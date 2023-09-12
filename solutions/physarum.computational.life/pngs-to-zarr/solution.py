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

    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    image_files.sort()  # sort filenames to maintain order

    # Load the first image to get shape and data type
    sample_img = imageio.imread(os.path.join(directory, image_files[0]))
    sample_img = np.moveaxis(sample_img, -1, 0)
    img_shape = sample_img.shape
    img_dtype = sample_img.dtype

    stack = np.zeros(
        tuple([len(image_files)] + list(img_shape)), dtype=img_dtype
    )

    print("Starting to read PNGs")
    for idx, image_file in enumerate(image_files):
        if idx % 10 == 0:
            print(f"Read {idx} of {len(image_files)}")
        img = imageio.imread(
            os.path.join(directory, image_file)
        )        
        stack[idx, :, :, :] = np.moveaxis(img, -1, 0)

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
    version="0.0.8",
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

