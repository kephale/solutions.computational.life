### album catalog: solutions.computational.life

from album.runner.api import setup, get_args
import time

def run():
    import numpy as np
    import pygfx as gfx
    import wgpu
    from wgpu.gui.auto import WgpuCanvas, run
    from scipy.spatial.transform import Rotation as R

    args = get_args()

    canvas_size = 800

    num_agents = args.num_agents if args.num_agents else 25
    max_speed = args.max_speed if args.max_speed else 0.1
    spring_constant = args.spring_constant if args.spring_constant else 0.5
    damping = args.damping if args.damping else 0.1
    boundary_size = args.boundary_size if args.boundary_size else 10.0
    gravity = args.gravity if args.gravity else 0.005  # Weaker gravity
    floor_elasticity = args.floor_elasticity if args.floor_elasticity else 0.5
    sphere_radius = 0.2

    # Initialize positions in a straight line
    positions = np.zeros((num_agents, 3))
    for i in range(num_agents):
        positions[i] = [i * 0.5, 0, 0]

    # Initialize velocities with a small initial velocity for the first agent
    velocities = np.zeros((num_agents, 3))
    velocities[0] = [max_speed, 0, 0]

    canvas = WgpuCanvas(title="Snake-like Simulation", size=(canvas_size, canvas_size))
    adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    device_wgpu = adapter.request_device()

    renderer = gfx.renderers.WgpuRenderer(canvas, show_fps=True)
    scene = gfx.Scene()
    camera = gfx.PerspectiveCamera(70, canvas_size / canvas_size)
    bounding_sphere = (0, 0, 0, boundary_size * 2)
    camera.show_object(bounding_sphere, view_dir=(0, 0, 1), up=(0, 1, 0), scale=1.0)
    controller = gfx.TrackballController(camera, register_events=renderer)

    geometry = gfx.sphere_geometry(radius=sphere_radius)
    material = gfx.MeshBasicMaterial(color=gfx.Color(0, 0, 1))
    meshes = [gfx.Mesh(geometry, material) for _ in range(num_agents)]

    for mesh in meshes:
        scene.add(mesh)

    box_material = gfx.MeshBasicMaterial(color=gfx.Color(1, 0, 0), wireframe=True)
    box_geometry = gfx.box_geometry(boundary_size * 2, boundary_size * 2, boundary_size * 2)
    bounding_box = gfx.Mesh(box_geometry, box_material)
    scene.add(bounding_box)

    floor_geometry = gfx.plane_geometry(boundary_size * 2, boundary_size * 2)
    floor_material = gfx.MeshBasicMaterial(color=gfx.Color(0.5, 0.5, 0.5))
    floor = gfx.Mesh(floor_geometry, floor_material)
    floor.local.position = (0, -boundary_size, 0)
    floor_rotation = R.from_euler('x', np.pi / 2).as_quat()
    floor.local.rotation = floor_rotation
    scene.add(floor)

    simulation_running = True

    def update_scene():
        for pos, mesh in zip(positions, meshes):
            mesh.local.position = pos
        renderer.render(scene, camera)

    def apply_spring_forces():
        nonlocal positions, velocities
        for i in range(1, num_agents):
            current_pos = positions[i]
            prev_pos = positions[i-1]
            displacement = current_pos - prev_pos
            distance = np.linalg.norm(displacement)
            direction = displacement / distance if distance != 0 else np.zeros(3)
            spring_force = -spring_constant * (distance - 0.5) * direction
            velocities[i] += spring_force - damping * velocities[i]
            velocities[i-1] -= spring_force - damping * velocities[i-1]

    def apply_joint_rotation():
        current_time = time.time()
        for i in range(1, num_agents):
            angle = np.sin(current_time + i * np.pi / num_agents) * 0.1
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            velocities[i] = np.dot(rotation_matrix, velocities[i])

    def clamp_velocities():
        nonlocal velocities
        speeds = np.linalg.norm(velocities, axis=1)
        too_fast = speeds > max_speed
        velocities[too_fast] *= max_speed / speeds[too_fast].reshape(-1, 1)

    def apply_gravity_and_floor():
        nonlocal velocities, positions
        for i in range(num_agents):
            velocities[i][1] -= gravity
            if positions[i][1] < -boundary_size + sphere_radius:
                positions[i][1] = -boundary_size + sphere_radius
                velocities[i][1] *= -floor_elasticity

    def animate():
        nonlocal positions
        if simulation_running:
            apply_spring_forces()
            apply_joint_rotation()
            clamp_velocities()
            apply_gravity_and_floor()
            positions += velocities
            for i in range(num_agents):
                for dim in range(3):
                    if positions[i][dim] > boundary_size:
                        velocities[i][dim] = -velocities[i][dim]
                        positions[i][dim] = boundary_size
                    elif positions[i][dim] < -boundary_size and dim != 1:
                        velocities[i][dim] = -velocities[i][dim]
                        positions[i][dim] = -boundary_size
            update_scene()
        canvas.request_draw()

    @renderer.add_event_handler("key_down")
    def on_key(event):
        nonlocal simulation_running, positions, velocities
        if event.key == "r":
            positions = np.zeros((num_agents, 3))
            for i in range(num_agents):
                positions[i] = [i * 0.5, 0, 0]
            velocities = np.zeros((num_agents, 3))
            velocities[0] = [max_speed, 0, 0]
        elif event.key == "p":
            simulation_running = False
        elif event.key == "s":
            simulation_running = True

    canvas.request_draw(animate)
    run()

setup(
    group="3d-simulations",
    name="snake-simulation",
    version="0.0.2",
    title="Snake-like Simulation using pygfx and NumPy",
    description="An album solution to run a snake-like simulation using pygfx and NumPy with Hooke-style springs, sinusoidal joint rotations, gravity, and floor interactions.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Kyle Harrington", "url": "https://kyleharrington.com"}],
    tags=["simulation", "pygfx", "NumPy", "Python", "snake", "spring", "gravity"],
    license="MIT",
    covers=[
        {
            "description": "Snake-like simulation example.",
            "source": "cover.png",
        }
    ],
    album_api_version="0.5.1",
    run=run,
    args=[
        {
            "name": "num_agents",
            "description": "Number of agents in the simulation",
            "type": "integer",
            "required": False,
        },
        {
            "name": "max_speed",
            "description": "Maximum speed of agents",
            "type": "float",
            "required": False,
        },
        {
            "name": "spring_constant",
            "description": "Spring constant for Hookes Law",
            "type": "float",
            "required": False,
        },
        {
            "name": "damping",
            "description": "Damping factor for the springs",
            "type": "float",
            "required": False,
        },
        {
            "name": "boundary_size",
            "description": "Size of the boundary for the simulation",
            "type": "float",
            "required": False,
        },
        {
            "name": "gravity",
            "description": "Gravitational force applied to the agents",
            "type": "float",
            "required": False,
        },
        {
            "name": "floor_elasticity",
            "description": "Elasticity of the floor when agents collide with it",
            "type": "float",
            "required": False,
        },
    ],
    dependencies={
        "parent": {
            "group": "environments",
            "name": "physical-simulation",
            "version": "0.0.2"
        }
    }
)
