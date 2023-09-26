from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

env_file = StringIO(
    """channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - zarr
  - numpy
  - Pillow
  - pip
"""
)


def scan_image_and_append_to_zarr(zarr_path, device, resolution=300):
    from io import StringIO
    import subprocess
    import time
    from datetime import datetime
    import zarr
    import numpy as np
    from PIL import Image
    import tempfile
    import os
    
    # Get the current date and time in a specific format
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', dir=".", delete=False) as temp:
        output_file = temp.name
        cmd = [
            "scanimage",
            "--format",
            "png",
            "-d",
            device,
            "--resolution",
            str(resolution),
            "--mode",
            "Color",
            "--output-file",
            output_file,
        ]

        # Execute the command
        start_time = time.time()  # Note the start time
        print(f"Running: {cmd}")
        subprocess.run(cmd)
        
        # Open the scanned image
        image = Image.open(output_file)
        arr = np.array(image)

        # Open the Zarr store
        store = zarr.NestedDirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=False)

        # If dataset doesn't exist, create it with an extensible first dimension
        if 'images' not in root:
            zarr_array = root.zeros('images', shape=(0, *arr.shape), dtype=arr.dtype, chunks=(1, 1024, 1024, 3), 
                                    compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE))
        else:
            zarr_array = root['images']

        # Append new image to the Zarr array
        zarr_array.resize(tuple([zarr_array.shape[0] + 1] + list(zarr_array.shape[1:])))
        zarr_array[-1] = arr

        print(f"Number of nonzeros in arr: {np.count_nonzero(arr)}")

        print(f"New zarr shape: {zarr_array.shape}")
        
        elapsed_time = time.time() - start_time  # Calculate elapsed time

        # Remove the temporary file
        os.remove(output_file)

        return {"elapsed_time": elapsed_time, "zarr_file": zarr_path}

def run():
    from album.runner.api import get_args
    import time

    timestep = int(get_args().timestep)
    try:
        while True:
            res = scan_image_and_append_to_zarr(get_args().zarr_path, get_args().device)
            time_taken = res["elapsed_time"]
            sleep_time = max(timestep - time_taken, 0)  # Ensure non-negative sleep time
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
        
setup(
    group="physarum.computational.life",
    name="scan-as-zarr",
    version="0.0.6",
    title="Scan images as zarr.",
    description="An Album solution for scanning a timeseries into a zarr file.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "scanning", "acquisition", "Python", "zarr"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {
            "name": "zarr_path",
            "type": "string",
            "description": "Zarr path for saving results.",
            "required": True,
        },
        {
            "name": "device",
            "type": "string",
            "description": "Device id for scanimage to use",
            "required": True,
        },
        {
            "name": "timestep",
            "type": "string",
            "description": "Timestep for scanning",
            "required": True,
            "default": "120",
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
