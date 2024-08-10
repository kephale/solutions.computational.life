###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import napari
    import numpy as np
    from perlin_noise import PerlinNoise
    from scipy.ndimage import gaussian_filter
    from qtpy.QtCore import QTimer

    # Constants
    size = 50
    noise_scale = 0.05
    height_scale = 10
    time_scale = 0.01
    noise = PerlinNoise(octaves=8)

    # Pre-compute meshgrid to avoid repeated calculations
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = x.flatten()
    y = y.flatten()

    # Function to generate terrain using Perlin noise
    def generate_terrain(time):
        terrain = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                terrain[i, j] = noise([i * noise_scale, j * noise_scale, time * time_scale]) * height_scale
        terrain = gaussian_filter(terrain, sigma=2)  # Apply Gaussian filter for extra smoothing
        return terrain

    # Function to generate vertices and faces for the surface
    def generate_surface(terrain):
        # Generate vertices
        z = terrain.flatten()
        vertices = np.column_stack([x, y, z])

        # Generate faces (two triangles per grid square)
        faces = []
        for i in range(size - 1):
            for j in range(size - 1):
                idx = i * size + j
                faces.append([idx, idx + 1, idx + size])
                faces.append([idx + 1, idx + size + 1, idx + size])
        faces = np.array(faces)

        return vertices, faces

    # Function to update the terrain dynamically
    def update_terrain(layer, time):
        terrain = generate_terrain(time)
        vertices, faces = generate_surface(terrain)
        layer.data = (vertices, faces, np.ones(vertices.shape[0]))
        layer.refresh()

    # Initialize Napari viewer
    viewer = napari.Viewer(ndisplay=3)
    initial_time = 0
    initial_terrain = generate_terrain(initial_time)
    vertices, faces = generate_surface(initial_terrain)
    layer = viewer.add_surface((vertices, faces, np.ones(vertices.shape[0])), name='3D Terrain')

    # Set the camera to the desired whole number configuration
    viewer.camera.center = (24, 24, 0.15)
    viewer.camera.zoom = 9
    viewer.camera.angles = (107, 38, -142)
    viewer.camera.perspective = 0

    # Timer to update the terrain surface
    timer = QTimer()
    def on_timeout():
        nonlocal initial_time
        initial_time += 1
        update_terrain(layer, initial_time)

    timer.timeout.connect(on_timeout)
    timer.start(200)

    napari.run()

setup(
    group="terrain",
    name="dynamic-terrain",
    version="0.0.2",
    title="3D Terrain Visualization with Perlin Noise",
    description="An Album solution that visualizes 3D terrain using Perlin noise and displays it as a dynamically changing surface using napari.",
    authors=["Kyle Harrington"],
    cite=[{
        "text": "Perlin, K., 1985. An image synthesizer. ACM SIGGRAPH Computer Graphics, 19(3), pp.287-296."
    }],
    tags=["Perlin Noise", "3D Terrain", "napari", "visualization"],
    license="MIT",
    covers=[{
        "description": "3D Terrain Visualization cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "napari",
            "version": "0.0.6"
        }
    }
)
