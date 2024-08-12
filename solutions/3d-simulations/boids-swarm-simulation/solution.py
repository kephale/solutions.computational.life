###album catalog: solutions.computational.life

from album.runner.api import setup, get_args

def run():
    import pybullet as p
    import pybullet_data
    import numpy as np
    import pygfx as gfx
    import wgpu
    from wgpu.gui.auto import WgpuCanvas, run

    args = get_args()

    canvas_size = 800

    num_boids = args.num_boids if args.num_boids else 100
    max_speed = args.max_speed if args.max_speed else 0.1
    max_force = args.max_force if args.max_force else 0.03
    separation_dist = args.separation_dist if args.separation_dist else 1.5
    alignment_dist = args.alignment_dist if args.alignment_dist else 2.0
    cohesion_dist = args.cohesion_dist if args.cohesion_dist else 2.0
    boundary_size = args.boundary_size if args.boundary_size else 10.0

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
            if np.linalg.norm(vel) > 1e-6:  # Avoid zero or near-zero velocities
                target_pos = pos + vel
                mesh.look_at(target_pos)
        renderer.render(scene, camera)

    p.connect(p.DIRECT)
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    boid_ids = []
    for i in range(num_boids):
        boid_id = p.loadURDF("sphere2.urdf", basePosition=positions[i].tolist())
        boid_ids.append(boid_id)

    def apply_boids_rules():
        nonlocal positions, velocities
        for i, boid_id in enumerate(boid_ids):
            aabbMin, aabbMax = p.getAABB(boid_id)
            neighbors = p.getOverlappingObjects(aabbMin, aabbMax)
            separation_force = np.zeros(3)
            alignment_force = np.zeros(3)
            cohesion_force = np.zeros(3)
            alignment_count = 0
            cohesion_count = 0

            for neighbor in neighbors:
                neighbor_id = neighbor[0]
                if neighbor_id != boid_id:
                    neighbor_pos = np.array(p.getBasePositionAndOrientation(neighbor_id)[0])
                    neighbor_vel = np.array(p.getBaseVelocity(neighbor_id)[0])
                    distance = np.linalg.norm(positions[i] - neighbor_pos)

                    if distance < separation_dist:
                        separation_force += (positions[i] - neighbor_pos)

                    if distance < alignment_dist:
                        alignment_force += neighbor_vel
                        alignment_count += 1

                    if distance < cohesion_dist:
                        cohesion_force += neighbor_pos
                        cohesion_count += 1

            separation_force = np.clip(separation_force, -max_force, max_force)

            if alignment_count > 0:
                alignment_force /= alignment_count
                alignment_force = np.clip(alignment_force - velocities[i], -max_force, max_force)

            if cohesion_count > 0:
                cohesion_force /= cohesion_count
                cohesion_force = np.clip(cohesion_force - positions[i], -max_force, max_force)

            velocities[i] += separation_force + alignment_force + cohesion_force
            velocities[i] = np.clip(velocities[i], -max_speed, max_speed)

    def apply_boundary_conditions():
        nonlocal positions, velocities
        for i in range(num_boids):
            for dim in range(3):
                if positions[i][dim] > boundary_size:
                    velocities[i][dim] = -velocities[i][dim]
                    positions[i][dim] = boundary_size
                elif positions[i][dim] < -boundary_size:
                    velocities[i][dim] = -velocities[i][dim]
                    positions[i][dim] = -boundary_size

    def animate():
        nonlocal positions
        if simulation_running:
            apply_boids_rules()
            apply_boundary_conditions()
            for i, boid_id in enumerate(boid_ids):
                positions[i] += velocities[i]
                p.resetBasePositionAndOrientation(boid_id, positions[i].tolist(), [0, 0, 0, 1])
                p.resetBaseVelocity(boid_id, linearVelocity=velocities[i].tolist())
            update_scene()
        canvas.request_draw()

    @renderer.add_event_handler("key_down")
    def on_key(event):
        nonlocal simulation_running, positions, velocities
        if event.key == "r":
            positions = np.random.rand(num_boids, 3) * (2 * boundary_size) - boundary_size
            velocities = (np.random.rand(num_boids, 3) * 2 - 1) * max_speed
            for i, boid_id in enumerate(boid_ids):
                p.resetBasePositionAndOrientation(boid_id, positions[i].tolist(), [0, 0, 0, 1])
                p.resetBaseVelocity(boid_id, linearVelocity=velocities[i].tolist())
        elif event.key == "p":
            simulation_running = False
        elif event.key == "s":
            simulation_running = True

    canvas.request_draw(animate)
    run()

setup(
    group="3d-simulations",
    name="boids-swarm-simulation",
    version="0.0.5",
    title="Boids Swarm Simulation using pygfx and pybullet",
    description="An album solution to run a Boids swarm simulation using pygfx and pybullet.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Boids Algorithm", "url": "https://en.wikipedia.org/wiki/Boids"}],
    tags=["simulation", "pygfx", "pybullet", "Python", "Boids", "swarm"],
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
            "version": "0.0.3"
        }
    }
)
