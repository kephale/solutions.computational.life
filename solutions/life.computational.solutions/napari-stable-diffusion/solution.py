from io import StringIO

from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

def run():
    import napari
    from skimage.io import imread
    from album.runner.api import get_args

    prompt = get_args().prompt
    print(prompt)
    napari.run()


setup(
    group="life.computational.solutions",
    name="napari-stable-diffusion",
    version="0.0.1-SNAPSHOT",
    title="napari-stable-diffusion",
    description="Open napari-stable-diffusion.",
    solution_creators=["Kyle Harrington"],
    cite=[],
    tags=["Python", "napari", "stable diffusion"],
    license="MIT",
    covers=[{
        "description": "Open napari-stable-diffusion.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[{
        "name": "prompt",
        "type": "string",
        "description": "The prompt to render with napari-stable-diffusion",
        "required": True,
    }],
    run=run,
    dependencies={
        "parent": {
            "group": "life.computational.solutions",
            "name": "napari",
            "version": "0.0.1-SNAPSHOT"
        }
    },
)
