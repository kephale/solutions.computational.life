###album catalog: solutions.computational.life

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

    # Neural Cellular Automata (NCA) model definition
    class NCA(nn.Module):
        def __init__(self, channel_n=16, fire_rate=0.5):
            super(NCA, self).__init__()
            self.channel_n = channel_n
            self.fire_rate = fire_rate
            self.perceive = nn.Conv2d(channel_n, channel_n, 3, padding=1)
            self.update = nn.Sequential(
                nn.Conv2d(channel_n, 128, 1),
                nn.ReLU(),
                nn.Conv2d(128, channel_n, 1)
            )

        def forward(self, x):
            dx = self.perceive(x)
            dx = self.update(dx)
            update_mask = (torch.rand(dx.shape[0], 1, dx.shape[2], dx.shape[3]) < self.fire_rate).float()
            x = x + dx * update_mask
            return x

    # Initialize the NCA
    nca = NCA().cuda()
    optimizer = torch.optim.Adam(nca.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    def train(nca, optimizer, loss_fn, steps=1000):
        for step in range(steps):
            x = torch.randn(1, 16, 64, 64).cuda()
            target = torch.randn(1, 16, 64, 64).cuda()
            optimizer.zero_grad()
            output = nca(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f'Step {step}, Loss: {loss.item()}')
        return output

    # Run the NCA and record output
    output = train(nca, optimizer, loss_fn)
    output_np = output.cpu().detach().numpy()
    
    # Create a video from NCA output
    video_path = f"{video_name}.avi"
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (64, 64))
    
    for frame in output_np[0]:
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
    name="nca-train-run",
    version="0.0.1",
    title="NCA Train and Run with Dropbox",
    description="An Album solution that trains and runs a neural cellular automata, saving output video to Dropbox.",
    authors=["Kyle Harrington"],
    cite=[],
    tags=["NCA", "python", "dropbox"],
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
            "default": "nca_output",
            "description": "Name of the output video file"
        }
    ],
    run=run,
    dependencies={'environment_file': env_file}
)
