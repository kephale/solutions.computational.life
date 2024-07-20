from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

env_file = StringIO(
    """channels:
  - pytorch-nightly
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - zarr
  - numpy
  - imageio
  - ome-zarr
  - opencv
  - napari
  - pip
  - dask
  - pandas
  - scipy
  - pyqt
  - matplotlib
  - xarray
  - hdf5
  - omero-py
  - pytorch
  - torchvision
  - diffusers
  - einops
  - pillow
  - openjpeg
  - imagecodecs
  - fftw
  - s3fs
  - pooch
  - qtpy
  - superqt
  - yappi
  - imageio
  - tqdm
  - tifffile
  - flake8
  - h5py
  - mypy
  - pint
  - opencv
  - flask
  - vtk
  - libnetcdf
  - ruff
  - lxml
  - jupyter
  - notebook
  - pytables
  - ipywidgets
  - meshio
  - pip:
      - "git+https://github.com/napari/napari.git"
      - blik
      - ipython
      - black
      - pre-commit
      - album
      - transformers
      - tensorstore
      - pydantic-ome-ngff
      - python-dotenv
      - ndjson
      - segment-anything
      - snakeviz
      - cryohub
      - cryoet-data-portal
      - napari-cryoet-data-portal
      - starfile
      - imodmodel
      - cryotypes
      - "git+https://github.com/teamtomo/membrain-seg.git"
"""
)


def run():
    pass


setup(
    group="cryoet.kyleharrington.com",
    name="cryoet-napari",
    version="0.0.1",
    title="Parent environment for some napari cryoet tools.",
    description="A parent environment for cryoet solutions",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington.", "url": "https://kyleharrington.com"}],
    tags=["imaging", "cryoet", "Python"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={"environment_file": env_file},
)
