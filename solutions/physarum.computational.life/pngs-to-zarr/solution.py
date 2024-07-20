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

    def extract_region(img, circle, max_radius):
        x, y, r = circle
        top, bottom = max(0, y-max_radius), min(img.shape[0], y+max_radius)
        left, right = max(0, x-max_radius), min(img.shape[1], x+max_radius)

        region = np.zeros((2*max_radius, 2*max_radius, 3), dtype=np.uint8)

        target_shape = region[max_radius-(y-top):max_radius+(bottom-y), max_radius-(x-left):max_radius+(right-x)].shape
        source_shape = img[top:bottom, left:right].shape

        if target_shape != source_shape:
            print(f"Shape mismatch: target={target_shape} vs source={source_shape}")
            print(f"Circle: {circle}")
            return region

        region[max_radius-(y-top):max_radius+(bottom-y), max_radius-(x-left):max_radius+(right-x)] = img[top:bottom, left:right]

        return region

    
    # Process each image
    for idx, filename in enumerate(image_files):
        print(f"Processing {idx} of {len(image_files)}")
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
        circles = find_circles(img)

        if circles is not None:
            max_radius = int(img.shape[1] / 4)  # Assuming the max radius as per find_circles
            for i, circle in enumerate(circles[0, :]):
                cropped_img = extract_region(img, circle, max_radius)
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
