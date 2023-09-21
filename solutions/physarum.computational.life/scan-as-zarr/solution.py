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
  - pip:
    - python-sane
"""
)



def scan_image_to_zarr(zarr_path, device, resolution=300):
    import subprocess
    import time
    from datetime import datetime

    import sane
    import numpy as np
    import zarr
    from PIL import Image

    
    start_time = time.time()
    
    # Initialize sane
    sane.init()

    # Open specified device
    dev = sane.open(device)

    # Set options
    params = dev.get_parameters()
    try:
        dev.depth = 8  # 8-bit depth
        dev.mode = 'color'  # color mode
        dev.resolution = resolution  # resolution
    except Exception as e:
        print(f"Failed to set scan settings: {e}")
        return

    # Start the scan and get a numpy array
    dev.start()
    arr = dev.arr_snap()

    # Check and adjust data type for consistency
    if arr.dtype == np.uint16:
        arr = (arr / 255).astype(np.uint8)

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
    zarr_array.resize(zarr_array.shape[0] + 1, axis=0)
    zarr_array[-1] = arr
        
    # Close the scanning device
    dev.close()

    # Return the path to the Zarr file and elapsed time
    elapsed_time = time.time() - start_time
    return {"elapsed_time": elapsed_time, "zarr_file": zarr_path}


def run():
    from album.runner.api import get_args

    import time
    
    timestep = int(get_args().timestep)
    try:
        while True:
            res = scan_image_to_zarr(get_args().zarr_path, get_args().device)
            time_taken = res["elapsed_time"]
            sleep_time = max(
                timestep - time_taken, 0
            )  # Ensure non-negative sleep time
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")


setup(
    group="physarum.computational.life",
    name="scan-as-zarr",
    version="0.0.2",
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
