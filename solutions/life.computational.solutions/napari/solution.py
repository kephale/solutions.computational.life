from io import StringIO

from album.runner.api import setup

def run():
    import napari
    from skimage.io import imread
    from album.runner.api import get_args

    napari.run()


setup(
    group="life.computational.solutions",
    name="napari",
    version="0.0.2",
    title="Open napari",
    description="Open napari.",
    solution_creators=["Kyle Harrington"],
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
    dependencies={
        "parent": {
            "group": "environments",
            "name": "napari",
            "version": "0.0.2"
        }
    }
)
