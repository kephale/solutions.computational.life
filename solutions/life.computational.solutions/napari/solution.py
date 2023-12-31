from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

env_file = StringIO("""channels:
- conda-forge
- pytorch-nightly
- defaults
dependencies:
- python=3.10
- qtpy
- "setuptools=59.5.0"
# issue with cmake from conda on macos 13?
#- cmake
- pybind11
- pip
- boost-cpp
- mpfr
- gmp
- cgal
- numpy
- scyjava >= 1.8.1
- scipy
- matplotlib
- pandas
- pytables
- jupyter
- notebook
- quantities
- ipywidgets
- pythreejs
- ipyvolume
#- vispy
- meshio
- zarr
- xarray
- hdf5
- mpfr
- gmp
- pyvista
- pyqt
- omero-py
- pyimagej >= 1.4.0
- pyopencl
- reikna
- openjdk=11
- jupyterlab
- pytorch
- torchvision
- diffusers
- einops
- fire
- maven
- pillow
- openjpeg
- imagecodecs
- "bokeh>=2.4.2,<3"
- python-graphviz
- ipycytoscape
- fftw
- napari-segment-blobs-and-things-with-membranes
- s3fs
- pooch
- qtpy
- superqt
- yappi
- ftfy
- tqdm
- imageio
- pyarrow
- squidpy
- h5py
- tifffile
- nilearn
- flake8
- pytest
- asv
- pint
- pytest-qt
- pytest-cov
- mypy
- opencv
- flask
- vedo
- vtk
- libnetcdf
- ruff
- qscintilla2
- confuse
- jpype1 >= 1.4.1
- labeling >= 0.1.12
- lazy_loader
- pip:
    - "--editable=git+git@github.com:zarr-developers/zarr-python.git"
    - "--editable=git+https://github.com/vispy/vispy.git"
    - "--editable=git+git@github.com:kephale/napari.git"
    - idr-py
    - album
    - omero-rois
    - imageio-ffmpeg
    - transformers
    - gradio
    - imaris-ims-file-reader
    - scanpy
    - pyarrow
    - invagination
    - hypothesis
    - tensorstore
    - alabaster
    - compressed-segmentation
    - pyspng-seunglab
    - tabulous
    - imglyb
    - imglyb-bdv
    - fibsem_tools
    - pyheif
    - "ome-zarr>=0.3.0"
    - importmagic
    - epc
    - ruff
    - python-lsp-server[all]
    - pylsp-mypy
    - pyls-isort
    - python-lsp-black
    - pyls-memestra
    - pylsp-rope
    - python-lsp-ruff
    - snakeviz
    - pyaudio
    - Mastodon.py
    - qrcode
    - napari-process-points-and-surfaces
    - opencv-python-headless
    - pygeodesic
    - skan
    - napari-boids
    - napari-matplotlib
    - stardist-napari
    - cellpose-napari
    - stardist
    - "tensorflow-macos;  platform_system==\"Darwin\" and platform_machine==\"arm64\""
    - "tensorflow-metal;  platform_system==\"Darwin\" and platform_machine==\"arm64\""
    - pydantic-ome-ngff
    - python-dotenv
    - validate-pyproject[all]
    - segment-anything
    - "--editable=git+git@github.com:napari/napari.git"
    - "--editable=git+https://github.com/tlambert03/napari-omero.git"
    - "--editable=git+https://github.com/napari/napari-console.git"
    - "--editable=git+https://github.com/napari/napari-animation.git"
    - "--editable=git+https://github.com/CBI-PITT/napari-imaris-loader"
    - "--editable=git+https://github.com/leoguignard/napari-turing"
    - "--editable=git+git@github.com:kephale/napari-stable-diffusion.git"
    - "--editable=git+https://github.com/DamCB/tyssue"
    - "--editable=git+https://github.com/DamCB/tyssue-demo"
    - "--editable=git+https://github.com/DamCB/tyssue-notebooks"
    - "--editable=git+git@github.com:kephale/napari-tyssue.git"
    - "--editable=git+git@github.com:seung-lab/corgie.git"
    - "--editable=git+git@github.com:kephale/napari-hierarchical.git"
    - "--editable=git+https://github.com/ome/napari-ome-zarr.git"
    - "--editable=git+git@github.com:andy-sweet/napari-metadata.git"
    - "--editable=git+https://github.com/jo-mueller/napari-stl-exporter.git"
    - "--editable=git+https://github.com/kephale/tootapari.git"
    - "--editable=git+https://github.com/letmaik/pyvirtualcam"
    - "--editable=git+git@github.com:kephale/napari-qrcode.git"
    - "--editable=git+git@github.com:kephale/pulser.git"
    - "--editable=git+git@github.com:JaneliaSciComp/KLEIO-Python-SDK.git"
    - "--editable=git+git@github.com:kephale/napari-kleio"
    - "--editable=git+git@github.com:napari/napari-graph.git"
    - "--editable=git+git@github.com:kephale/napari-conference.git"
    - "--editable=git+git@github.com:haesleinhuepf/napari-blender-bridge.git"
    - "--editable=git+git@github.com:imagej/napari-imagej.git"
    - "--editable=git+git@github.com:kephale/napari-workshop-browser.git"
    - "--editable=git+git@github.com:kephale/napari-workshop-template.git"
    - "--editable=git+git+https://github.com/morphometrics/morphometrics-engine.git"
    - napari-stress
    - napari-pymeshlab
    - "--editable=git+git+https://github.com/kevinyamauchi/morphometrics.git"
    - napari-skimage-regionprops
    - napari-geojson
    - napari-plot-profile
    - redlionfish
    - btrack
    - "--editable=git+git+https://github.com/lowe-lab-ucl/arboretum.git"
    - "--editable=git+git+https://github.com/hanjinliu/napari-spreadsheet.git"
    - "--editable=git+git+https://github.com/jacopoabramo/napari-live-recording.git"
    - "--editable=git+git+https://github.com/JoOkuma/napari-segment-anything.git"
    - natari
""")


def run():
    import napari
    from skimage.io import imread
    from album.runner.api import get_args

    napari.run()


setup(
    group="life.computational.solutions",
    name="napari",
    version="0.0.1-SNAPSHOT",
    title="Open napari",
    description="Open napari.",
    solution_creators=["Kyle Harrington", "Deborah Schmidt"],
    cite=[{
        "text": "napari contributors (2019). napari: a multi-dimensional image viewer for python.",
        "doi": "10.5281/zenodo.3555620",
        "url": "https://github.com/napari/napari"
    }],
    tags=["Python", "napari"],
    license="MIT",
    covers=[{
        "description": "Open the napari viewer.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={'environment_file': env_file}
)
