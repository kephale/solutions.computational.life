from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/


def find_circles(image, viewer=None):
    import cv2
    import numpy as np

    # 2. Pre-process the image (if necessary)
    # Convert the image to grayscale if it's not
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image
    )

    # Use median blur to reduce noise
    gray = cv2.medianBlur(gray, 5)

    # 3. Hough circle detection
    # The parameters may need adjustment based on your specific image
    minRadius = int(gray.shape[1] / 8)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minRadius,
        param1=50,
        param2=60,
        minRadius=minRadius,
        maxRadius=int(gray.shape[1] / 4),
    )

    # If some circles are detected, add them as a shapes layer to the viewer
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circle_data = [
            [(x, y), (r, r)] for (x, y, r) in circles
        ]  # adjusted format
        if viewer:
            viewer.add_shapes(
                circle_data,
                shape_type='ellipse',
                edge_color='green',
                edge_width=2,
            )

    return circles


def view(zarr_path):
    """
    Read a Zarr array.

    Args:
        zarr_path (str): Path to the Zarr array where images will be stored.
    """
    import os
    import numpy as np
    import zarr

    import napari

    print(f"Reading {zarr_path}")

    # Load Zarr data
    zarr_array = zarr.open(zarr_path, mode='r')

    # Open with napari
    viewer = napari.Viewer()

    # TODO is there a NGFF place to find the image name?
    viewer.add_image(zarr_array, name=zarr_path)

    timestep = int(zarr_array.shape[0] / 2)

    image = np.transpose(zarr_array[timestep, :, :, :], (1, 0, 2))
    image = np.mean(image, axis=2).astype(zarr_array.dtype)

    find_circles(image, viewer=viewer)


def run():
    from album.runner.api import get_args

    view(get_args().zarr_path)


setup(
    group="physarum.computational.life",
    name="view-physarum-napari",
    version="0.0.1",
    title="Open a Physarum zarr in napari.",
    description="An Album solution for viewing a Physarum zarr",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "zarr", "Python", "napari", "visualization", "Physarum"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
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
            "version": "0.0.3",
        }
    },
)

if __name__ == "__main__":
    zarr_path = "/Users/kharrington/Data/Physarum/experiment_005.zarr"
    view(zarr_path)
