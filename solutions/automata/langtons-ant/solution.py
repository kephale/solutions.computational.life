###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import napari
    import numpy as np
    from qtpy.QtCore import QTimer

    # Constants
    size = 200
    ant_position = [size // 2, size // 2]
    ant_direction = 0  # 0=up, 1=right, 2=down, 3=left
    ant_color = [255, 0, 0]  # Red color for the ant

    # Create a grid with two states (0=white, 1=black)
    grid = np.zeros((size, size), dtype=np.uint8)

    # Directions: up, right, down, left (dy, dx)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Function to update the ant's position
    def update_ant():
        nonlocal ant_position, ant_direction, grid

        # Determine the current cell state
        current_cell_state = grid[ant_position[0], ant_position[1]]

        # Flip the cell state
        grid[ant_position[0], ant_position[1]] = 1 - current_cell_state

        # Turn the ant: right on white (0), left on black (1)
        if current_cell_state == 0:
            ant_direction = (ant_direction + 1) % 4
        else:
            ant_direction = (ant_direction - 1) % 4

        # Move the ant forward
        ant_position[0] = (ant_position[0] + directions[ant_direction][0]) % size
        ant_position[1] = (ant_position[1] + directions[ant_direction][1]) % size

    # Update function for the Napari viewer
    def update_layer(layer, grid):
        update_ant()
        
        # Create an RGB image to highlight the ant
        rgb_grid = np.stack([grid * 255] * 3, axis=-1)  # Convert grid to RGB
        rgb_grid[ant_position[0], ant_position[1]] = ant_color  # Color the ant
        layer.data = rgb_grid
        layer.refresh()

    # Initialize Napari viewer
    viewer = napari.Viewer()
    layer = viewer.add_image(np.stack([grid] * 3, axis=-1), name='Langton\'s Ant')

    # Timer for updating the image layer
    timer = QTimer()
    timer.timeout.connect(lambda: update_layer(layer, grid))
    timer.start(50)  # Update every 50 ms

    napari.run()

setup(
    group="automata",
    name="langtons-ant",
    version="0.0.2",
    title="Langton's Ant Simulation",
    description="An Album solution that simulates Langton's Ant and displays it using napari.",
    authors=["Kyle Harrington"],
    cite=[{
        "text": "Langton, C.G., 1986. Studying artificial life with cellular automata. Physica D: nonlinear phenomena, 22(1-3), pp.120-149."
    }],
    tags=["Langton's Ant", "automata", "napari", "simulation"],
    license="MIT",
    covers=[{
        "description": "Langton's Ant Simulation cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "napari",
            "version": "0.0.7"
        }
    }
)
