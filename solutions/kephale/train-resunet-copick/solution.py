### album catalog: cellcanvas

from album.runner.api import get_args, setup

def run():
    import logging
    import sys
    import argparse

    import torch
    import pytorch_lightning as pl
    from monai.data import DataLoader
    from monai.transforms import (
        Compose, RandAffined, RandFlipd, RandRotate90d, EnsureChannelFirstd,
        ScaleIntensityd, Resized, ToTensord, AsDiscrete, EnsureType
    )
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    import torch.nn.functional as F

    from morphospaces.datasets import CopickDataset
    from morphospaces.transforms.label import LabelsAsFloat32
    from morphospaces.transforms.image import ExpandDimsd, StandardizeImage    
    from monai.networks.blocks import ResidualUnit
    from monai.networks.layers.factories import Act, Norm
    from monai.networks.nets import UNet    
    from torch.nn import CrossEntropyLoss

    from copick_torch import data, transforms, training, log_setup
    import mlflow

    args = get_args()

    copick_config_path = args.copick_config_path
    train_run_names = args.train_run_names
    val_run_names = args.val_run_names
    tomo_type = args.tomo_type
    user_id = args.user_id
    session_id = args.session_id
    segmentation_type = args.segmentation_type
    voxel_spacing = args.voxel_spacing
    lr = args.lr
    logdir = args.logdir
    experiment_name = args.experiment_name
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    num_res_units = args.num_res_units

    # setup logging
    log = log_setup.setup_logging()

    # patch parameters
    patch_shape = (96, 96, 96)
    patch_stride = (96, 96, 96)
    patch_threshold = 0.5

    image_key = "zarr_tomogram"
    labels_key = "zarr_mask"

    logdir_path = "./" + logdir

    pl.seed_everything(42, workers=True)

    train_transform = transforms.get_train_transform(image_key, labels_key)
    val_transform = transforms.get_val_transform(image_key, labels_key)

    train_ds, unique_train_label_values = data.load_dataset(
        copick_config_path, train_run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, train_transform, patch_shape, patch_stride, labels_key, patch_threshold
    )

    val_ds, unique_val_label_values = data.load_dataset(
        copick_config_path, val_run_names, tomo_type, user_id, session_id, segmentation_type, voxel_spacing, val_transform, patch_shape, patch_stride, labels_key, patch_threshold
    )

    unique_label_values = set(unique_train_label_values).union(set(unique_val_label_values))
    num_classes = len(unique_label_values)
    in_channels = num_classes  # Set the number of in_channels to the number of unique labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    class ResUNetSegmentation(pl.LightningModule):
        def __init__(self, lr, num_classes, in_channels, num_res_units):
            super().__init__()
            self.lr = lr
            self.num_classes = num_classes
            self.in_channels = in_channels
            self.num_res_units = num_res_units
            self.model = self.build_model(num_classes, in_channels, num_res_units)
            self.loss_function = CrossEntropyLoss()
            self.val_outputs = []

        def build_model(self, num_classes, in_channels, num_res_units):
            return torch.nn.Sequential(
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=in_channels,
                    out_channels=16,
                    strides=1,
                    norm=Norm.BATCH,
                    act=Act.PRELU,
                ),
                UNet(
                    spatial_dims=3,
                    in_channels=16,
                    out_channels=num_classes,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=num_res_units,
                )
            )

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.squeeze(1).long()  # Convert labels to Long and squeeze
            outputs = self.forward(images)
            loss = self.loss_function(outputs, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            mlflow.log_metric("train_loss", loss.item(), step=self.global_step)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch[image_key], batch[labels_key]
            labels = labels.squeeze(1).long()  # Convert labels to Long and squeeze
            outputs = self.forward(images)

            # Debugging information
            if batch_idx == 0:
                text_log = {
                    "Debug/Images Shape": str(images.shape),
                    "Debug/Labels Shape": str(labels.shape),
                    "Debug/Outputs Shape": str(outputs.shape),
                    "Debug/Labels Unique Values": str(torch.unique(labels).tolist()),
                    "Debug/Outputs Unique Values": str(torch.unique(outputs).tolist())
                }
                mlflow.log_dict(text_log, "validation_debug_info.json")

            try:
                val_loss = self.loss_function(outputs, labels)
                self.log("val_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                mlflow.log_metric("val_loss", val_loss.item(), step=self.global_step)
            except RuntimeError as e:
                print(f"Validation loss computation failed: {e}")
                print(f"Output shape: {outputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Output unique values: {torch.unique(outputs)}")
                print(f"Label unique values: {torch.unique(labels)}")
                raise e

            self.val_outputs.append(val_loss)
            return val_loss

        def on_validation_epoch_end(self):
            self.val_outputs.clear()

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

        
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    net = ResUNetSegmentation(lr=lr, num_classes=num_classes, in_channels=in_channels, num_res_units=num_res_units)

    training.train_model(net, train_loader, val_loader, lr, logdir_path, 100, 0.15, 4, max_epochs=max_epochs, model_name="resunet", experiment_name=experiment_name)

setup(
    group="kephale",
    name="train-resunet-copick",
    version="0.0.11",
    title="Train 3D ResUNet for Segmentation with Copick Dataset",
    description="Train a 3D ResUNet network using the Copick dataset for segmentation.",
    solution_creators=["Kyle Harrington", "Zhuowen Zhao"],
    cite=[{"text": "Morphospaces team.", "url": "https://github.com/morphometrics/morphospaces"}],
    tags=["imaging", "segmentation", "cryoet", "Python", "morphospaces"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "copick_config_path",
            "description": "Path to the Copick configuration file",
            "type": "string",
            "required": True,
        },
        {
            "name": "train_run_names",
            "description": "Names of the runs in the Copick project for training",
            "type": "string",
            "required": True,
        },
        {
            "name": "val_run_names",
            "description": "Names of the runs in the Copick project for validation",
            "type": "string",
            "required": True,
        },
        {
            "name": "tomo_type",
            "description": "Tomogram type in the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "user_id",
            "description": "User ID for the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "session_id",
            "description": "Session ID for the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "segmentation_type",
            "description": "Segmentation type in the Copick project",
            "type": "string",
            "required": True,
        },
        {
            "name": "voxel_spacing",
            "description": "Voxel spacing for the Copick project",
            "type": "float",
            "required": True,
        },
        {
            "name": "lr",
            "description": "Learning rate for the ResUNet training",
            "type": "float",
            "required": False,
            "default": 0.0001
        },
        {
            "name": "logdir",
            "description": "Output directory name in the current working directory. Default is checkpoints",
            "type": "string",
            "required": False,
            "default": "checkpoints",
        },
        {
            "name": "experiment_name",
            "description": "mlflow experiment name. Default is resunet_experiment",
            "type": "string",
            "required": False,
            "default": "resunet_experiment",
        },  
        {
            "name": "max_epochs",
            "description": "Maximum number of epochs for training",
            "type": "integer",
            "required": False,
            "default": 10000
        },
        {
            "name": "batch_size",
            "description": "Batch size for training and validation",
            "type": "integer",
            "required": False,
            "default": 1
        },
        {
            "name": "num_res_units",
            "description": "Number of residual units in the UNet",
            "type": "integer",
            "required": False,
            "default": 2
        }                        
    ],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "copick-monai",
            "version": "0.0.2"
        }
    }
)
