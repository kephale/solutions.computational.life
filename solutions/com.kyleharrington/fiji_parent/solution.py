from album.runner.api import setup

# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

env_file = """channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pyimagej
  - openjdk=11.0.9.1
  - jgo
"""

def init_ij():
    import imagej
    # see this link for initialization options: https://github.com/imagej/pyimagej/blob/master/doc/Initialization.md
    return imagej.init(['net.imagej:imagej'])


def install():
    print("Downloading maven dependencies. This may take a while. "
          "By default, maven dependencies are shared between applications - "
          "installing another solution with similar dependencies should be fast.")
    init_ij()

    
def run():
    pass


setup(
    group="com.kyleharrington",
    name="fiji_parent",
    version="0.1.1",
    title="Java-based software parent solution",
    description="Parent of Java-based software solutions: Fiji, ImageJ, ImgLib2, etc.",
    authors=["Kyle Harrington"],
    cite=[{
        "text": "SciJava community.",
        "url": "https://github.com/scijava"
    }],
    tags=["java", "fiji", "bigdataviewer", "imagej", "scijava"],
    license="MIT",
    documentation=[],
    covers=[{
        "description": "Cover image for java-based software.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[],
    install=install,
    run=run,
    dependencies={'environment_file': env_file}
)


if __name__== "__main__":
    run()
