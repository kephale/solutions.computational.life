import os
from album.runner.api import setup, get_data_path

def local_repository_path():
    if not os.path.exists(get_data_path()):
        os.makedirs(get_data_path())
    
    return os.path.join(get_data_path(), "git")


def install():
    import subprocess
        
    repo_url = "https://github.com/scenerygraphics/sciview"
    clone_path = local_repository_path()
    
    # Check if the repo already exists, if not, clone it
    if not os.path.exists(os.path.join(clone_path, ".git")):
        subprocess.check_call(["git", "clone", repo_url, clone_path])

    os.chdir(local_repository_path())
        
    # Use subprocess to run ./gradlew build
    build_cmd = os.path.join(clone_path, "gradlew")
    subprocess.check_call([build_cmd, "build"])

def run():
    import subprocess

    os.chdir(local_repository_path())
    
    # run with ./gradlew runImageJMain
    clone_path = local_repository_path()
    run_cmd = os.path.join(clone_path, "gradlew")
    subprocess.check_call([run_cmd, "runImageJMain"])


setup(
    group="sc.iview",
    name="sciview",
    version="0.1.1",
    title="sciview",
    description="sciview is a 3D/VR/AR visualization tool for large data from the Fiji community",
    authors=["Kyle Harrington"],
    cite=[{
        "text": "sciview team.",
        "url": "https://github.com/scenerygraphics/sciview"
    }],
    tags=["sciview", "fiji", "imagej", "3d", "vr", "ar"],
    license="MIT",
    documentation=[],
    covers=[{
        "description": "Cover image for sciview. Shows a schematic of the sciview UI with an electron microscopy image that contains segmented neurons.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[],
    install=install,
    run=run,
    dependencies={
        "parent": {
            "group": "com.kyleharrington",
            "name": "fiji_parent",
            "version": "0.1.1",
        }
    },
)


# if __name__== "__main__":
#     def get_data_path():
#         return "/tmp/album_test"
    
#     install()
#     run()
