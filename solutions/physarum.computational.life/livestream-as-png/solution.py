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
  - pip
  - Pillow
  - pip:
    - Mastodon.py
    - python-dotenv
    - appdirs
"""
)

import subprocess
import time
from datetime import datetime


def scan_image(directory, device, mastodon, resolution=300):
    from PIL import Image
    import os
    
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

    # After scanning, downscale the image to reduce its size
    temp_output_file = f"{directory}/temp_physarum_{current_datetime}.png"

    with Image.open(output_file) as img:
        # A common approach is to resize while maintaining the aspect ratio
        # Here, we halve the dimensions. Adjust as needed.
        width, height = img.size
        new_width = width // 2
        new_height = height // 2
        img_rescaled = img.resize((new_width, new_height), Image.ANTIALIAS)
        
        # Save the downscaled image to the temp file
        img_rescaled.save(temp_output_file, "PNG")

    # Done scanning, now toot
    
    description = "An image of the slime mold Physarum polycephalum. This is part of a sequence of toots. This caption is autogenerated as is the toot."

    text = f"A livestream from phsarum.computational.life. This is timestep {time.time()}."
    
    # Post the image to Mastodon
    media_metadata = mastodon.media_post(
        temp_output_file, mime_type="image/png", description=description
    )
    
    # Send toot with the attached image
    mastodon.status_post(text, media_ids=media_metadata["id"])

    os.remove(temp_output_file)
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    
    return {"elapsed_time": elapsed_time, "output-file": output_file}


def run():
    from album.runner.api import get_args

    import os
    from mastodon import Mastodon

    from dotenv import load_dotenv

    from pathlib import Path
    from appdirs import user_config_dir

    # Load environment variables
    load_dotenv(Path(user_config_dir("tootapari", "kephale")) / ".env")
    
    mastodon = Mastodon(
        access_token=os.getenv("MASTODON_ACCESS_TOKEN"),
        api_base_url=os.getenv("MASTODON_INSTANCE_URL"),
    )

    if not mastodon:
        print("Cannot login to mastodon")
        return
    
    timestep = int(get_args().timestep)
    try:
        while True:
            res = scan_image(get_args().output_directory, get_args().device, mastodon)
            time_taken = res["elapsed_time"]
            sleep_time = max(timestep - time_taken, 0)  # Ensure non-negative sleep time
            time.sleep(
                sleep_time
            )  # Adjust sleep time based on time taken by the subprocess
    except KeyboardInterrupt:
        print("\nScript terminated by user.")


setup(
    group="physarum.computational.life",
    name="livestream-as-png",
    version="0.0.1",
    title="Livestream images as PNGs.",
    description="An Album solution for scanning images into PNG files in a directory and live tooting.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "scanning", "acquisition", "Python", "mastodon"],
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
