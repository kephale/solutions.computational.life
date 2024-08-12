###album catalog: solutions.computational.life

from io import StringIO
from album.runner.api import setup
import os
import subprocess

env_file = StringIO(
    """name: alife-visualization
channels:
  - conda-forge
  - pytorch
dependencies:
  - python==3.10
  - pip
  - numpy<2
  - jupyter
  - ipywidgets
  - scipy
  - matplotlib
  - pandas
  - seaborn
  - scikit-learn
  - sympy
  - networkx
  - pillow
  - numba
  - h5py
  - dask
  - scikit-image
  - napari
  - pyside2
  - zarr
  - xarray
  - pillow
  - pooch
  - h5py
  - pint
  - fastapi
  - trimesh
  - tensorstore
  - pip:
    - imageio[ffmpeg]
    - torch>=2.0.0
    - opencv-python
    - pyGLM
    - noise
    - git+https://github.com/kephale/napari-screen-recorder.git
    - Mastodon.py
    - ndjson
    - mrcfile
    - perlin-noise
    - networkx
    - album
    - git+https://github.com/kephale/tootapari.git
"""
)

def run():
    # Get the path to the conda environment
    conda_prefix = os.getenv("CONDA_PREFIX")
    
    # Detect the user's default shell
    shell_command = os.getenv("SHELL", "/bin/bash")  # Default to /bin/bash if SHELL is not set
    
    # If the environment is activated, launch the shell
    if conda_prefix:
        print(f"Entering the system's shell in the environment: {conda_prefix}")
        subprocess.run(shell_command, check=False)
    else:
        print("Conda environment is not activated. Please activate the environment first.")

setup(
    group="environments",
    name="napari",
    version="0.0.7",
    title="An environment for visualizing artificial life simulations",
    description="An album solution that provides a generalized environment for visualizing alife simulations using napari.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington", "url": "https://kyleharrington.com"}],
    tags=["simulation", "visualization", "Python", "environment", "napari"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    dependencies={"environment_file": env_file},
    run=run
)
