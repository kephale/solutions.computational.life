###album catalog: solutions.computational.life

from album.runner.api import setup


def read_images_to_zarr(directory, zarr_path):
    """
    Read RGB PNG images from a directory, detect petri dishes in each image, and store each timepoint
    for each petri dish in a separate Zarr group.

    Args:
        directory (str): Path to the directory containing the images.
        zarr_path (str): Path to the Zarr array where images will be stored.
    """
    import os
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import re
    import zarr
    
    print(f"Reading from {directory} and writing to {zarr_path}")

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    image_files.sort()  # sort filenames to maintain order

    # Open the Zarr store
    store = zarr.NestedDirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    def extract_timestamp(filename):
        """
        Extract timestamp from the filename.
        Assumes filename format like 'physarum_YYYYMMDD_HHMMSS.png'.
        """
        mc = re.search(r'(\d{8}_\d{6})', filename)
        return mc.group(0) if mc else None
    
    def detect_circles(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            return np.uint16(np.around(circles))
        return None
    
    # Process each image
    for filename in image_files:
        timestamp = extract_timestamp(filename)
        if not timestamp:
            print(f"Timestamp not found in filename {filename}, skipping.")
            continue

        img_path = os.path.join(directory, filename)
        if not os.path.isfile(img_path):
            print(f"File not found or is not a file: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        circles = detect_circles(img)

        if circles is not None:
            for i, circle in enumerate(circles[0, :]):
                x, y, r = circle[0], circle[1], circle[2]
                cropped_img = img[y-r:y+r, x-r:x+r]
                cropped_img = np.moveaxis(cropped_img, -1, 0)

                # Create a unique group name using dish index and timestamp
                group_name = f"petri_dish_{i}_{timestamp}"
                dish_group = root.require_group(group_name)

                # Store the cropped image in the group
                dataset_name = f"timepoint_{timestamp}"
                dish_group.array(dataset_name, cropped_img, chunks=(1024, 1024, 3), dtype=cropped_img.dtype,
                                 compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.SHUFFLE))

    print(f"All petri dishes stored in {zarr_path}")


def run():
    from album.runner.api import get_args

    read_images_to_zarr(get_args().png_directory, get_args().zarr_path)


setup(
    group="physarum.computational.life",
    name="pngs-to-zarr",
    version="0.1.2",
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
            "version": "0.0.4",
        }
    },
)

# if True:
#     png_directory = "/Users/kharrington/Data/Physarum/experiment_004_mini"
#     zarr_path = "/Users/kharrington/Data/Physarum/experiment_004_mini_v2.zarr"
#     read_images_to_zarr(png_directory, zarr_path)
