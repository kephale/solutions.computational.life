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


def run():
    pass


setup(
    group="physarum.computational.life",
    name="parent-environment",
    version="0.0.1",
    title="Parent environment for physarum.computational.life.",
    description="A parent environment for physarum.computational.life solutions",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "png", "zarr", "Python"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={"environment_file": env_file},
)
