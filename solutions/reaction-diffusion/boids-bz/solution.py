###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import napari
    import torch
    import numpy as np
    import time
    from superqt.utils import ensure_main_thread, thread_worker
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
    from qtpy.QtCore import Qt

    # Detect device for torch
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Parameters for the BZ reaction
    alpha, beta, gamma = 1.0, 1.0, 1.0

    # Boid parameters
    num_boids = 100
    boundary_size = 20
    max_speed = 0.1
    max_force = 0.03
    separation_dist = 1.5
    alignment_dist = 2.0
    cohesion_dist = 2.0

    class SimulationState:
        def __init__(self):
            self.paused = False
            self.worker = None

        def toggle_pause(self):
            self.paused = not self.paused

        def is_paused(self):
            return self.paused

    def initialize_reaction(size, device):
        return torch.rand((2, 3, *size), device=device)

    def update_reaction(p, arr, alpha, beta, gamma):
        q = (p + 1) % 2
        with torch.no_grad():
            padded_arr = torch.nn.functional.pad(arr[p], (1, 1, 1, 1, 1, 1), mode='replicate')
            s = torch.zeros((3, *arr.shape[2:]), device=arr.device)
            for k in range(3):
                s[k] = (
                    padded_arr[k, :-2, 1:-1, 1:-1] +
                    padded_arr[k, 2:, 1:-1, 1:-1] +
                    padded_arr[k, 1:-1, :-2, 1:-1] +
                    padded_arr[k, 1:-1, 2:, 1:-1] +
                    padded_arr[k, 1:-1, 1:-1, :-2] +
                    padded_arr[k, 1:-1, 1:-1, 2:] +
                    padded_arr[k, 1:-1, 1:-1, 1:-1]
                ) / 7
            arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
            arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
            arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])
            arr[q] = torch.clamp(arr[q], 0, 1)

        return arr, q

    def initialize_boids(num_boids, boundary_size, max_speed):
        positions = np.random.rand(num_boids, 3) * (2 * boundary_size) - boundary_size
        velocities = (np.random.rand(num_boids, 3) * 2 - 1) * max_speed
        return positions, velocities

    def apply_boids_rules(positions, velocities):
        for i in range(num_boids):
            separation_force = np.zeros(3)
            alignment_force = np.zeros(3)
            cohesion_force = np.zeros(3)
            alignment_count = 0
            cohesion_count = 0

            for j in range(num_boids):
                if i != j:
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                    if distance < separation_dist:
                        separation_force += diff / distance
                    if distance < alignment_dist:
                        alignment_force += velocities[j]
                        alignment_count += 1
                    if distance < cohesion_dist:
                        cohesion_force += positions[j]
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
        return velocities

    def update_boids(positions, velocities, arr):
        velocities = apply_boids_rules(positions, velocities)
        scaling_factor = arr.shape[2] / (2 * boundary_size)
        for i in range(num_boids):
            positions[i] += velocities[i]
            x, y, z = ((positions[i] + boundary_size) * scaling_factor).astype(int)
            if 0 <= x < arr.shape[2] and 0 <= y < arr.shape[3] and 0 <= z < arr.shape[4]:
                arr[0, 2, x, y, z] += 0.1
            positions[i] = np.clip(positions[i], -boundary_size, boundary_size)
            if np.any(np.abs(positions[i]) >= boundary_size):
                velocities[i] = -velocities[i]
        
        return positions, velocities

    @thread_worker
    def frame_generator(arr, positions, velocities, state):
        p = 0
        while True:
            if not state.is_paused():
                arr, p = update_reaction(p, arr, alpha, beta, gamma)
                positions, velocities = update_boids(positions, velocities, arr)
                yield arr[p].cpu().numpy(), positions
            time.sleep(0.05)

    size = (20, 20, 20)
    arr = initialize_reaction(size, device)
    positions, velocities = initialize_boids(num_boids, boundary_size, max_speed)

    viewer = napari.Viewer(ndisplay=3)
    u_layer = viewer.add_image(np.zeros(size), name="BZ Field U", rendering='mip')
    points_layer = viewer.add_points(positions, size=1, face_color="blue", name="Boids")

    @ensure_main_thread
    def update_layers(args):
        frame, boid_positions = args
        u_layer.data = frame[0]
        points_layer.data = (boid_positions + boundary_size) * (arr.shape[2] / (2 * boundary_size))

    state = SimulationState()
    worker = frame_generator(arr, positions, velocities, state)
    worker.yielded.connect(update_layers)
    worker.start()

    napari.run()

setup(
    group="reaction-diffusion",
    name="boids-bz",
    version="0.0.2",
    title="Boids with BZ Reaction Trail",
    description="Simulates a boids swarm painting trails in a Belousov-Zhabotinsky reaction field.",
    solution_creators=["Kyle Harrington"],
    tags=["reaction-diffusion", "BZ reaction", "boids", "napari"],
    license="MIT",
    album_api_version="0.5.1",
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "physical-simulation",
            "version": "0.0.10"
        }
    }
)
