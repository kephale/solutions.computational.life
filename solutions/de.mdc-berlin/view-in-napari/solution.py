from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

env_file = StringIO("""channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - scikit-image=0.19.1
  - napari=0.4.16
""")


def run():
    import napari
    from skimage.io import imread
    from album.runner.api import get_args

    img = imread(get_args().input)
    napari.view_image(img, rgb=False, name="Input image")
    napari.run()


setup(
    group="de.mdc-berlin",
    name="view-in-napari",
    version="0.1.0",
    title="View a dataset in napari v0.4.16",
    description="An Album solution for displaying a dataset with napari.",
    solution_creators=["Deborah Schmidt"],
    cite=[{
        "text": "napari contributors (2019). napari: a multi-dimensional image viewer for python.",
        "doi": "10.5281/zenodo.3555620",
        "url": "https://github.com/napari/napari"
    }],
    tags=["template", "napari"],
    license="MIT",
    covers=[{
        "description": "A dataset displayed in napari.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[{
        "name": "input",
        "type": "file",
        "description": "The image about to be displayed in napari.",
        "required": True
    }],
    run=run,
    dependencies={'environment_file': env_file}
)
