###album catalog: solutions.computational.life

from io import StringIO
from album.runner.api import setup

env_file = StringIO(
    """name: game-of-life
channels:
  - conda-forge
dependencies:
  - python>=3.10
  - pip
  - pygfx
  - numpy
  - jupyter
  - ipywidgets
  - pip:
    - pyside6
    - glfw
    - imageio[ffmpeg]
"""
)

def run(canvas_size=800):
    import pygfx as gfx
    import numpy as np
    from wgpu.gui.auto import WgpuCanvas, run

    def initialize_grid(size):
        return np.random.choice([0, 1], size=size)

    def update_grid(grid):
        neighbors = sum(np.roll(np.roll(grid, i, 0), j, 1)
                        for i in (-1, 0, 1) for j in (-1, 0, 1)
                        if (i != 0 or j != 0))
        return (neighbors == 3) | ((grid == 1) & (neighbors == 2))

    def main():
        grid_size = 100
        grid = initialize_grid((grid_size, grid_size))

        canvas = WgpuCanvas(size=(canvas_size, canvas_size))
        renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
        scene = gfx.Scene()
        camera = gfx.OrthographicCamera(canvas_size, canvas_size)
        camera.local.y = canvas_size / 2
        camera.local.scale_y = -1
        camera.local.x = canvas_size / 2
        controller = gfx.PanZoomController(camera, register_events=renderer)

        material = gfx.MeshBasicMaterial(color="#FFFFFF")
        geometry = gfx.plane_geometry(1, 1)
        meshes = []

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                mesh = gfx.Mesh(geometry, material)
                mesh.world.position = (i - grid_size // 2, j - grid_size // 2, 0)
                row.append(mesh)
                scene.add(mesh)
            meshes.append(row)

        def update_scene():
            nonlocal grid
            grid = update_grid(grid)
            for i in range(grid_size):
                for j in range(grid_size):
                    meshes[i][j].visible = bool(grid[i, j])
            renderer.render(scene, camera)

        def animate():
            update_scene()
            canvas.request_draw()

        canvas.request_draw(animate)
        run()

    main()

setup(
    group="cellular-automata",
    name="game-of-life",
    version="0.0.2",
    title="Game of Life Simulation using pygfx",
    description="An album solution to run a Game of Life simulation using pygfx with zoom functionality.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Conway's Game of Life", "url": "https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life"}],
    tags=["simulation", "pygfx", "Python", "Game of Life"],
    license="MIT",
    covers=[
        {
            "description": "Game of Life simulation example.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    run=run,
    dependencies={"environment_file": env_file},
)
