###album catalog: solutions.computational.life

from io import StringIO
from album.runner.api import setup
import tempfile
import os

env_file = StringIO(
    """name: boids-swarm-simulation
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
  - pybullet
  - napari
  - pip:
    - pyside6
    - glfw
    - imageio[ffmpeg]
    - torch>=2.0.0
    - opencv-python
    - pyGLM
    - git+https://github.com/pygfx/pygfx
    - git+https://github.com/pygfx/wgpu-py.git@14e84e1956c608c592061f44a619430f0a2d7aa0
    - mediapipe
"""
)

setup(
    group="environments",
    name="physical-simulation",
    version="0.0.9",
    title="An environment to support multiple physical visualized artificial life simulations",
    description="An album solution that contains a generalized environment for alife simulations.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington", "url": "https://kyleharrington.com"}],
    tags=["simulation", "visualization", "Python", "environment"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    dependencies={"environment_file": env_file},
)
