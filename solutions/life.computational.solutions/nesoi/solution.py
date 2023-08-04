from io import StringIO


from album.runner.api import setup

def fetch_url(url):
    import requests
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        return response.text

    except requests.exceptions.RequestException as e:
        print(f"Error while fetching the URL: {e}")
        return None


# Please import additional modules at the beginning of your method declarations.
# More information: https://docs.album.solutions/en/latest/solution-development/

env_file = StringIO(fetch_url("https://raw.githubusercontent.com/kephale/nesoi/main/environment.yml"))

def install():
    bash_script = fetch_url("https://raw.githubusercontent.com/kephale/nesoi/main/create_environment")

    import subprocess

    try:
        subprocess.run(["bash", "-c", bash_script], check=True)
        print("Bash script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running the Bash script: {e}")


def run():
    import napari

    napari.run()


setup(
    group="life.computational.solutions",
    name="nesoi",
    version="0.0.1-SNAPSHOT",
    title="Open napari",
    description="Open napari in the nesoi environment.",
    solution_creators=["Kyle Harrington"],
    cite=[],
    tags=["Python"],
    license="MIT",
    covers=[{
        "description": "Open the napari viewer in nesoi environment.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={'environment_file': env_file}
)
