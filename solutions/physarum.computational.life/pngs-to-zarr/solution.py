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
  - imageio
"""
)

import subprocess
import time
from datetime import datetime

def process_image(idx_image_file_tuple, directory, z):
    idx, image_file = idx_image_file_tuple
    if idx % 10 == 0:
        print(f"Processed {idx} of {len(image_files)}")
    img = imageio.imread(os.path.join(directory, image_file))
    z[idx, ...] = img

def read_images_to_zarr(directory, zarr_path):
    """
    Read RGB PNG images from a directory and store them in a Zarr array.
    
    Args:
        directory (str): Path to the directory containing the images.
        zarr_path (str): Path to the Zarr array where images will be stored.
    """
    
    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    image_files.sort()  # sort filenames to maintain order
    
    # Load the first image to get shape and data type
    sample_img = imageio.imread(os.path.join(directory, image_files[0]))
    img_shape = sample_img.shape
    img_dtype = sample_img.dtype

    # Initialize the NestedDirectoryStore and Zarr array
    store = zarr.NestedDirectoryStore(zarr_path)
    z = zarr.open_array(store, mode='w', shape=(len(image_files),) + img_shape, dtype=img_dtype)

    # Parallelize
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_image, 
                     [(idx, image_file) for idx, image_file in enumerate(image_files)],
                     [directory] * len(image_files),
                     [z] * len(image_files))
    
    print(f"All images stored in {zarr_path}")


def run():
    import os
    import imageio
    
    from album.runner.api import get_args

    read_images_to_zarr(get_args().png_directory, get_args().zarr_path)


setup(
    group="physarum.computational.life",
    name="pngs-to-zarr",
    version="0.0.1",
    title="Convert PNGs to zarr.",
    description="An Album solution for converting a directory of PNGs into a zarr",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "png", "zarr", "Python"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {
            "name": "png_directory",
            "type": "string",
            "description": "Directory for PNG images.",
            "required": True,
        },
        {
            "name": "zarr_path",
            "type": "string",
            "description": "Path for saving zarrs",
            "required": True,
        },
    ],
    run=run,
    dependencies={"environment_file": env_file},
)
