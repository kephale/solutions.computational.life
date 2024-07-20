###album catalog: solutions.computational.life

from album.runner.api import setup, get_args

def run():
    import pygfx as gfx
    import numpy as np
    import wgpu
    from wgpu.gui.auto import WgpuCanvas, run
    from scipy.spatial import KDTree
    from numba import njit

    args = get_args()

    canvas_size = 800

    num_boids = args.num_boids if args.num_boids else 100
    max_speed = args.max_speed if args.max_speed else 0.1
    max_force = args.max_force if args.max_force else 0.03
    separation_dist = args.separation_dist if args.separation_dist else 1.5
    alignment_dist = args.alignment_dist if args.alignment_dist else 2.0
    cohesion_dist = args.cohesion_dist if args.cohesion_dist else 2.0
    boundary_size = args.boundary_size if args.boundary_size else 10.0
    neighborhood_radius = max(separation_dist, alignment_dist, cohesion_dist) * 1.5

    positions = np.random.rand(num_boids, 3) * (2 * boundary_size) - boundary_size
    velocities = (np.random.rand(num_boids, 3) * 2 - 1) * max_speed

    canvas = WgpuCanvas(title="Boids Swarm Simulation", size=(canvas_size, canvas_size))
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device_wgpu = adapter.request_device()

    renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, canvas_size / canvas_size)
    bounding_sphere = (0, 0, 0, boundary_size * 2)
    camera.show_object(bounding_sphere, view_dir=(0, 0, 1), up=(0, 1, 0), scale=1.0)
    controller = gfx.TrackballController(camera, register_events=renderer)

    geometry = gfx.cylinder_geometry(radius_top=0.0, radius_bottom=0.2, height=1.0, radial_segments=8)
    material = gfx.MeshBasicMaterial(color=gfx.Color(0, 0, 1))
    meshes = [gfx.Mesh(geometry, material) for _ in range(num_boids)]

    for mesh in meshes:
        scene.add(mesh)

    box_material = gfx.MeshBasicMaterial(color=gfx.Color(1, 0, 0), wireframe=True)
    box_geometry = gfx.box_geometry(boundary_size * 2, boundary_size * 2, boundary_size * 2)
    bounding_box = gfx.Mesh(box_geometry, box_material)
    scene.add(bounding_box)

    simulation_running = True

    def update_scene():
        for pos, vel, mesh in zip(positions, velocities, meshes):
            mesh.local.position = pos
            target_pos = pos + vel
            mesh.look_at(target_pos)
        renderer.render(scene, camera)

    @njit
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm

    @njit
    def apply_boids_rules(positions, velocities, neighbors, neighbors_len, num_boids, max_force, max_speed, separation_dist, alignment_dist, cohesion_dist):
        for i in range(num_boids):
            separation_force = np.zeros(3)
            alignment_force = np.zeros(3)
            cohesion_force = np.zeros(3)
            alignment_count = 0
            cohesion_count = 0

            for j in range(neighbors_len[i]):
                neighbor_idx = neighbors[i, j]
                if i != neighbor_idx:
                    distance = np.linalg.norm(positions[i] - positions[neighbor_idx])

                    if distance < separation_dist:
                        separation_force += (positions[i] - positions[neighbor_idx])
                    
                    if distance < alignment_dist:
                        alignment_force += velocities[neighbor_idx]
                        alignment_count += 1

                    if distance < cohesion_dist:
                        cohesion_force += positions[neighbor_idx]
                        cohesion_count += 1

            separation_force = normalize(separation_force) * max_force

            if alignment_count > 0:
                alignment_force /= alignment_count
                alignment_force = normalize(alignment_force) * max_speed
                alignment_force -= velocities[i]
                alignment_force = normalize(alignment_force) * max_force

            if cohesion_count > 0:
                cohesion_force /= cohesion_count
                cohesion_force = normalize(cohesion_force - positions[i]) * max_speed
                cohesion_force -= velocities[i]
                cohesion_force = normalize(cohesion_force) * max_force

            velocities[i] += separation_force + alignment_force + cohesion_force
            velocities[i] = normalize(velocities[i]) * max_speed

    @njit
    def apply_boundary_conditions(positions, velocities, num_boids, boundary_size):
        for i in range(num_boids):
            for dim in range(3):
                if positions[i][dim] > boundary_size:
                    velocities[i][dim] = -velocities[i][dim]
                    positions[i][dim] = boundary_size
                elif positions[i][dim] < -boundary_size:
                    velocities[i][dim] = -velocities[i][dim]
                    positions[i][dim] = -boundary_size

    def convert_neighbors(neighbors, num_boids):
        max_len = max(len(neigh) for neigh in neighbors)
        neighbors_array = np.zeros((num_boids, max_len), dtype=np.int32)
        neighbors_len = np.zeros(num_boids, dtype=np.int32)
        for i in range(num_boids):
            for j, neighbor in enumerate(neighbors[i]):
                neighbors_array[i, j] = neighbor
            neighbors_len[i] = len(neighbors[i])
        return neighbors_array, neighbors_len

    def animate():
        nonlocal positions, velocities
        if simulation_running:
            kdtree = KDTree(positions)
            neighbors = kdtree.query_ball_tree(kdtree, neighborhood_radius)
            neighbors, neighbors_len = convert_neighbors(neighbors, num_boids)

            apply_boids_rules(positions, velocities, neighbors, neighbors_len, num_boids, max_force, max_speed, separation_dist, alignment_dist, cohesion_dist)
            apply_boundary_conditions(positions, velocities, num_boids, boundary_size)
            positions += velocities
            update_scene()
        canvas.request_draw()

    @renderer.add_event_handler("key_down")
    def on_key(event):
        nonlocal simulation_running, positions, velocities
        if event.key == "r":
            positions = np.random.rand(num_boids, 3) * (2 * boundary_size) - boundary_size
            velocities = (np.random.rand(num_boids, 3) * 2 - 1) * max_speed
        elif event.key == "p":
            simulation_running = False
        elif event.key == "s":
            simulation_running = True

    canvas.request_draw(animate)
    run()

setup(
    group="3d-simulations",
    name="boids-swarm-simulation",
    version="0.0.2",
    title="Boids Swarm Simulation using pygfx",
    description="An album solution to run a Boids swarm simulation using pygfx.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Boids Algorithm", "url": "https://en.wikipedia.org/wiki/Boids"}],
    tags=["simulation", "pygfx", "Python", "Boids", "swarm"],
    license="MIT",
    covers=[
        {
            "description": "Boids swarm simulation example.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    run=run,
    args=[
        {
            "name": "num_boids",
            "description": "Number of boids in the simulation",
            "type": "integer",
            "required": False,
        },
        {
            "name": "max_speed",
            "description": "Maximum speed of boids",
            "type": "float",
            "required": False,
        },
        {
            "name": "max_force",
            "description": "Maximum force applied to boids",
            "type": "float",
            "required": False,
        },
        {
            "name": "separation_dist",
            "description": "Distance for separation between boids",
            "type": "float",
            "required": False,
        },
        {
            "name": "alignment_dist",
            "description": "Distance for alignment between boids",
            "type": "float",
            "required": False,
        },
        {
            "name": "cohesion_dist",
            "description": "Distance for cohesion between boids",
            "type": "float",
            "required": False,
        },
        {
            "name": "boundary_size",
            "description": "Size of the boundary for the simulation",
            "type": "float",
            "required": False,
        },
    ],
    dependencies={
        "parent": {
            "group": "environments",
            "name": "physical-simulation",
            "version": "0.0.1"
        }
    }
)
