###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

env_file = """channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - fsspec
  - dropboxdrivefs
"""


def run():
    import fsspec

    args = get_args()
    name = args.name
    token = args.token

    print(f"Hi {name}, nice to meet you!")

    # Using the DropboxFileSystem
    fs = fsspec.filesystem('dropbox', token=token)

    # Define the file path and content
    file_path = '/test_file.txt'
    content = 'Hello, Dropbox! This is a test file.'

    # Write the file to Dropbox
    with fs.open(file_path, 'w') as f:
        f.write(content)

    print(f'File written to {file_path} in Dropbox.')


setup(
    group="test",
    name="dropbox-write-test",
    version="0.0.1",
    title="Dropbox write test for fsspec",
    description="An Album solution testing a dropbox setup.",
    authors=["Kyle Harrington"],
    cite=[],
    tags=["template", "python"],
    license="unlicense",
    documentation=["documentation.md"],
    covers=[{
        "description": "Dummy cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[
        {
            "name": "name",
            "type": "string",
            "default": "Bugs Bunny",
            "description": "How do you want to be addressed?"
        },
        {
            "name": "token",
            "type": "string",
            "default": "",
            "description": "Dropbox access token"
        }
    ],
    run=run,
    dependencies={'environment_file': env_file}
)
