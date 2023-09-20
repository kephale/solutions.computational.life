from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

def read_images_to_zarr(directory, zarr_path):
    """
    Read RGB PNG images from a directory and store them in a Zarr array.

    Args:
        directory (str): Path to the directory containing the images.
        zarr_path (str): Path to the Zarr array where images will be stored.
    """
    import os
    import imageio.v2 as imageio
    import numpy as np
    import zarr

    import dask.array as da
    from dask import delayed

    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    print(f"Reading from {directory} and writing to {zarr_path}")
    
    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    image_files.sort()  # sort filenames to maintain order

    # Load the first image to get shape and data type
    sample_img = imageio.imread(os.path.join(directory, image_files[0]))
    sample_img = np.moveaxis(sample_img, -1, 0)
    img_shape = sample_img.shape
    img_dtype = sample_img.dtype

    # TODO read with dask delayed
    # Use dask.delayed to lazily read each image
    @delayed
    def read_image(filename, image_directory=directory):
        try:
            img = imageio.imread(
                os.path.join(image_directory, filename)
            )
            return np.moveaxis(img, -1, 0)
        except OSError as e:
            if "broken data stream" in str(e):  # Specifically handle this error
                print(f"Error reading image '{filename}': {e}")
            else:  # Handle any other OSErrors
                print(f"Unexpected error reading image '{filename}': {e}")
            return None

    # Create a list of delayed objects
    images_delayed = [read_image(f) for f in image_files]

    # Combine the delayed objects into a Dask array
    stack = da.stack([da.from_delayed(img, shape=img_shape, dtype=img_dtype) for img in images_delayed], axis=0)

    print(f"Stack shape {stack.shape} and type {stack.dtype}")
    
    print("Starting to write zarr")
    os.mkdir(zarr_path)
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=stack,
        group=root,
        axes="tcyx",
        storage_options=dict(chunks=img_shape),
    )

    print(f"All images stored in {zarr_path}")


def run():
    from album.runner.api import get_args

    read_images_to_zarr(get_args().png_directory, get_args().zarr_path)


setup(
    group="physarum.computational.life",
    name="pngs-to-zarr",
    version="0.0.10",
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
    dependencies={
        "parent": {
            "group": "physarum.computational.life",
            "name": "parent-environment",
            "version": "0.0.2",
        }
    },
)

# if True:
#     png_directory = "/Users/kharrington/Data/Physarum/experiment_004_mini"
#     zarr_path = "/Users/kharrington/Data/Physarum/experiment_004_mini_v2.zarr"
#     read_images_to_zarr(png_directory, zarr_path)
