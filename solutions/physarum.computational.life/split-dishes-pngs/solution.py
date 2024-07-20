###album catalog: solutions.computational.life

from album.runner.api import setup


def read_images_and_save_regions(source_directory, destination_directory):
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

    def extract_region_and_save(img, circle, max_radius, region_dir, timestamp, img_index, padding=10):
        x, y, r = circle
        # Increase the radius by the padding amount
        r_padded = r + padding

        top, bottom = max(0, y - r_padded), min(img.shape[0], y + r_padded)
        left, right = max(0, x - r_padded), min(img.shape[1], x + r_padded)

        region = np.zeros((2 * r_padded, 2 * r_padded, 3), dtype=np.uint8)

        target_shape = region[max_radius-(y-top):max_radius+(bottom-y), max_radius-(x-left):max_radius+(right-x)].shape
        source_shape = img[top:bottom, left:right].shape

        if target_shape != source_shape:
            print(f"Shape mismatch: target={target_shape} vs source={source_shape}")
            print(f"Circle: {circle}")
            return region

        region[max_radius-(y-top):max_radius+(bottom-y), max_radius-(x-left):max_radius+(right-x)] = img[top:bottom, left:right]

        padded_index = str(img_index).zfill(5)
        
        region_filename = os.path.join(region_dir, f"{timestamp}_{padded_index}.png")
        cv2.imwrite(region_filename, cv2.cvtColor(region, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving


    def match_circles(baseline_circles, new_circles):
        """
        Match new circles to baseline circles based on proximity.
        Returns a list of indices in new_circles that correspond to baseline_circles.
        """
        matches = [-1] * len(baseline_circles)
        for i, (xb, yb, rb) in enumerate(baseline_circles):
            min_dist = float('inf')
            matched_idx = -1
            for j, (xn, yn, rn) in enumerate(new_circles):
                dist = np.sqrt((xb - xn) ** 2 + (yb - yn) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    matched_idx = j
                    matches[i] = matched_idx
        return matches

    def find_extents_for_all_dishes(source_directory):
        listing = sorted([f for f in os.listdir(source_directory) if f.endswith('.png')])
        total_images = len(listing)
        extents = {}
        circles_cache = {}  # Cache for storing circles of each image

        def update_extents(circle, extents_index):
            x, y, r = circle
            left, top = x - r, y - r
            right, bottom = x + r, y + r

            if extents_index not in extents:
                extents[extents_index] = [left, top, right, bottom]
            else:
                extents[extents_index][0] = min(extents[extents_index][0], left)   # Min x
                extents[extents_index][1] = min(extents[extents_index][1], top)    # Min y
                extents[extents_index][2] = max(extents[extents_index][2], right)  # Max x
                extents[extents_index][3] = max(extents[extents_index][3], bottom) # Max y

        def get_circles(image_idx):
            if image_idx in circles_cache:
                return circles_cache[image_idx]
            img_path = os.path.join(source_directory, listing[image_idx])
            img = cv2.imread(img_path)

            # Check if the image was loaded successfully
            if img is None:
                print(f"Failed to load image: {img_path}")
                return None  # Return None if the image cannot be loaded

            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            circles = find_circles(img)
            circles_cache[image_idx] = circles
            return circles

        def categorize_and_sort(circles):
            y_values = [c[1] for c in circles]
            y_sorted_indices = np.argsort(y_values)

            # Split into 3 categories: top, middle, bottom
            top_indices = y_sorted_indices[:2]
            middle_indices = y_sorted_indices[2:4]
            bottom_indices = y_sorted_indices[4:]

            # Sort by x within each category
            top_indices = sorted(top_indices, key=lambda i: circles[i][0])
            middle_indices = sorted(middle_indices, key=lambda i: circles[i][0])
            bottom_indices = sorted(bottom_indices, key=lambda i: circles[i][0])

            # Combine the indices
            sorted_indices = top_indices + middle_indices + bottom_indices

            return [circles[i] for i in sorted_indices]

        
        def positions_same(start_idx, end_idx, tolerance=10):
            circles_start = get_circles(start_idx)
            circles_end = get_circles(end_idx)

            if circles_start is None or circles_end is None:
                return False

            circles_start = categorize_and_sort(get_circles(start_idx))
            circles_end = categorize_and_sort(get_circles(end_idx))

            print(f"positions_same\n{'_'.join([str(el) for el in circles_start])}\n{'_'.join([str(el) for el in circles_end])}")

            if len(circles_start) != len(circles_end):
                return False

            for (x_start, y_start, r_start), (x_end, y_end, r_end) in zip(circles_start, circles_end):
                if (abs(x_start - x_end) > tolerance or 
                    abs(y_start - y_end) > tolerance or 
                    abs(r_start - r_end) > tolerance):
                    return False
            return True

        def binary_search(start_idx, end_idx):
            print(f"Searching {start_idx} to {end_idx}")
            if start_idx >= end_idx:
                return

            if positions_same(start_idx, end_idx):
                circles_start_sorted = categorize_and_sort(get_circles(start_idx))
                for circle_index, circle in enumerate(circles_start_sorted):
                    update_extents(circle, circle_index)
            else:
                mid_idx = (start_idx + end_idx) // 2
                binary_search(start_idx, mid_idx)
                binary_search(mid_idx + 1, end_idx)

        binary_search(0, total_images - 1)
        return extents

    
    def crop_and_save(img, box, output_dir, timestamp, index):
        """
        Crop the image using the provided box and save it.
        """
        cropped_img = img[box[1]:box[3], box[0]:box[2]]
        padded_index = str(index).zfill(5)
        filename = os.path.join(output_dir, f"{timestamp}_{padded_index}.png")
        cv2.imwrite(filename, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

    
        
    print(f"Reading from {source_directory} and saving regions to {destination_directory}")

    baseline_circles = None
    extents = find_extents_for_all_dishes(source_directory)

    print(f"Extents:\n{extents}")
    
    for idx, filename in enumerate(sorted(os.listdir(source_directory))):
        if not filename.endswith('.png'):
            continue

        print(f"Processing {idx + 1} of {len(os.listdir(source_directory))}")
        timestamp = extract_timestamp(filename)
        if not timestamp:
            print(f"Timestamp not found in filename {filename}, skipping.")
            continue

        img_path = os.path.join(source_directory, filename)
        if not os.path.isfile(img_path):
            print(f"File not found or is not a file: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i, box in extents.items():
            region_dir = os.path.join(destination_directory, f"region_{i}")
            os.makedirs(region_dir, exist_ok=True)
            crop_and_save(img, box, region_dir, timestamp, idx)


    print(f"Regions saved in separate directories within {destination_directory}")



def run():
    from album.runner.api import get_args

    read_images_and_save_regions(get_args().input_directory, get_args().output_directory)


setup(
    group="physarum.computational.life",
    name="split-dishes-pngs",
    version="0.0.1",
    title="Split pngs into subdirs of cropped dishes.",
    description="Split pngs into subdirs of cropped dishes.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "png", "zarr", "Python"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {
            "name": "input_directory",
            "type": "string",
            "description": "Directory for input PNG images.",
            "required": True,
        },
        {
            "name": "output_directory",
            "type": "string",
            "description": "Directory for output PNG images.",
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
#     input_directory = "/Volumes/T7/data/physarum/experiment_001"
#     output_directory = "/tmp/experiment_001_cropped"
#     read_images_and_save_regions(input_directory, output_directory)
