from album.runner.api import setup, get_args


# Define the function to download PDB files
def download_pdbs(pdb_ids):
    import os
    import requests

    base_url = "https://files.rcsb.org/download/"
    for pdb_id in pdb_ids:
        pdb_url = f"{base_url}{pdb_id}.pdb"
        response = requests.get(pdb_url)
        if response.status_code == 200:
            with open(f"{pdb_id}.pdb", "wb") as file:
                file.write(response.content)
            print(f"Downloaded {pdb_id}")
        else:
            print(
                f"Failed to download {pdb_id}: HTTP Status Code {response.status_code}"
            )


def run():
    pdb_list = get_args().pdbs.split(
        ","
    )  # Assume the PDB IDs are passed as a comma-separated list
    download_pdbs(pdb_list)


# Set up the Album catalog entry
setup(
    group="kyleharrington",
    name="download-pdbs",
    version="0.0.1",
    title="PDB File Downloader",
    description="A utility to download PDB files from a list of PDB IDs.",
    solution_creators=["Your Name"],
    tags=["bioinformatics", "PDB", "protein structures", "data download"],
    license="MIT",
    album_api_version="0.5.1",
    args=[
        {
            "name": "pdbs",
            "type": "string",
            "description": "Comma-separated list of PDB IDs to download",
        }
    ],
    run=run,
)
