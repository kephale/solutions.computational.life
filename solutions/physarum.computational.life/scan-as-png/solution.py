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
"""
)

import subprocess
import time
from datetime import datetime


def scan_image(directory, device, resolution=300):
    # Get the current date and time in a specific format
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # The command to be run with the formatted date and time in the filename
    output_file = f"{directory}/physarum_{current_datetime}.png"
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
    elapsed_time = time.time() - start_time  # Calculate elapsed time

    return {"elapsed_time": elapsed_time, "output-file": output_file}


def run():
    from album.runner.api import get_args

    timestep = int(get_args().timestep)
    try:
        while True:
            res = scan_image(get_args().output_directory, get_args().device)
            time_taken = res["elapsed_time"]
            sleep_time = max(timestep - time_taken, 0)  # Ensure non-negative sleep time
            time.sleep(
                sleep_time
            )  # Adjust sleep time based on time taken by the subprocess
    except KeyboardInterrupt:
        print("\nScript terminated by user.")


setup(
    group="physarum.computational.life",
    name="scan-as-png",
    version="0.0.2",
    title="Scan images as PNGs.",
    description="An Album solution for scanning images into PNG files in a directory.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "scanning", "acquisition", "Python"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {
            "name": "output_directory",
            "type": "string",
            "description": "Directory for saving results.",
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
