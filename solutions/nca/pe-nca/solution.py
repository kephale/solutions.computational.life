from album.runner.api import get_args, setup

env_file = """channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pip
  - pip:
    - fsspec
    - dropboxdrivefs
    - dropbox
    - requests
    - numpy
    - torch
    - torchvision
    - matplotlib
    - opencv-python
"""

def run():
    import fsspec
    import torch
    import torch.nn as nn
    import numpy as np
    import cv2
    import os
    from torchvision import transforms

    args = get_args()
    token = args.token
    video_name = args.video_name

    # Define the DropboxFileSystem
    fs = fsspec.filesystem('dropbox', token=token)

    # Positional encoding function
    def add_positional_encoding(x):
        batch_size, channels, height, width = x.shape
        pos_enc = torch.zeros(batch_size, 2, height, width).to(x.device)
        for i in range(height):
            for j in range(width):
                pos_enc[:, 0, i, j] = i / height
                pos_enc[:, 1, i, j] = j / width
        return torch.cat([x, pos_enc], dim=1)

    # Neural Cellular Automata (NCA) model definition with positional encoding
    class NCA(nn.Module):
        def __init__(self, channel_n=16, fire_rate=0.5):
            super(NCA, self).__init__()
            self.channel_n = channel_n + 2  # Adjust channel number for positional encoding
            self.fire_rate = fire_rate
            self.perceive = nn.Conv2d(self.channel_n, channel_n, 3, padding=1)  # input channels increased
            self.update = nn.Sequential(
                nn.Conv2d(channel_n, 128, 1),
                nn.ReLU(),
                nn.Conv2d(128, channel_n, 1)
            )

        def forward(self, x):
            x = add_positional_encoding(x)
            dx = self.perceive(x)
            dx = self.update(dx)
            update_mask = (torch.rand(dx.shape[0], 1, dx.shape[2], dx.shape[3]).to(x.device) < self.fire_rate).float()
            x = x[:, :self.channel_n - 2, :, :]  # Ensure the size of x matches dx before addition
            x = x + dx * update_mask
            return x

    # Initialize the NCA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nca = NCA().to(device)
    optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)

    # Contrastive Loss function
    def contrastive_loss(output1, output2, margin=1.0):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-margin) * torch.pow(euclidean_distance, 2) +
                                      (margin) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

    # Training loop with contrastive loss
    def train(nca, optimizer, steps=1000, margin=1.0):
        for step in range(steps):
            x = torch.randn(1, 16, 64, 64).to(device)
            target = torch.randn(1, 16, 64, 64).to(device)
            optimizer.zero_grad()

            output1 = nca(x)
            output2 = nca(output1.detach())

            loss = contrastive_loss(output1, output2, margin)
            l2_reg = sum(param.pow(2.0).sum() for param in nca.parameters())
            loss += 1e-5 * l2_reg

            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f'Step {step}, Loss: {loss.item()}')
        return output1

    # Run the NCA and record output
    output = train(nca, optimizer, steps=1000)
    output_np = output.cpu().detach().numpy()

    # Create a video from NCA output
    video_path = f"{video_name}.avi"
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (64, 64))

    for _ in range(300):
        output = nca(output)
        frame = output.cpu().detach().numpy()[0]
        frame = frame.transpose(1, 2, 0)  # Ensure the frame has the shape (64, 64, 16)
        frame = np.mean(frame, axis=2)  # Reduce the channel dimension to 1 by averaging
        frame = ((frame - frame.min()) / (frame.max() - frame.min()) * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()

    # Upload the video to Dropbox
    dropbox_video_path = f'/videos/{video_name}.avi'
    with fs.open(dropbox_video_path, 'wb') as f:
        with open(video_path, 'rb') as video_file:
            f.write(video_file.read())

    print(f'Video saved to {dropbox_video_path} in Dropbox.')

setup(
    group="nca",
    name="pe-nca",
    version="0.0.2",
    title="NCA Train and Run with Positional Encoding and Dropbox",
    description="An Album solution that trains and runs a neural cellular automata with positional encoding, saving output video to Dropbox.",
    authors=["Kyle Harrington"],
    cite=[],
    tags=["NCA", "python", "dropbox", "positional encoding"],
    license="unlicense",
    documentation=["documentation.md"],
    covers=[{
        "description": "NCA output video cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[
        {
            "name": "token",
            "type": "string",
            "default": "",
            "description": "Dropbox access token"
        },
        {
            "name": "video_name",
            "type": "string",
            "default": "nca_output_pos_enc",
            "description": "Name of the output video file"
        }
    ],
    run=run,
    dependencies={'environment_file': env_file}
)
