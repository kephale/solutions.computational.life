###album catalog: solutions.computational.life

from io import StringIO
from album.runner.api import setup
import tempfile
import os

env_file = StringIO(
    """name: alife-visualization
channels:
  - conda-forge
  - defaults
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
  - pybullet
  - pip:
    - napari
    - imageio[ffmpeg]
    - torch>=2.0.0
    - opencv-python
    - pyGLM
"""
)

setup(
    group="environments",
    name="napari",
    version="0.0.1",
    title="An environment for visualizing artificial life simulations",
    description="An album solution that provides a generalized environment for visualizing alife simulations using napari.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington", "url": "https://kyleharrington.com"}],
    tags=["simulation", "visualization", "Python", "environment", "napari"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    dependencies={"environment_file": env_file},
)