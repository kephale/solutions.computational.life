###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import cv2
    import napari
    import numpy as np
    from superqt import ensure_main_thread
    from superqt.utils import thread_worker
    from scipy.spatial import KDTree
    import colorsys
    from qtpy.QtCore import QTimer
    import time

    class Boid:
        def __init__(self, position, velocity, radius=5):
            self.position = np.array(position, dtype=float)
            self.velocity = np.array(velocity, dtype=float)
            self.radius = radius

        def move(self, max_speed=5.0, frame_shape=None):
            speed = np.linalg.norm(self.velocity)
            if speed > max_speed:
                self.velocity = (self.velocity / speed) * max_speed
            self.position += self.velocity

            # Adjust bounds checks for resized frame
            if frame_shape is not None:
                self.position = np.clip(self.position, [0, 0], [frame_shape[1], frame_shape[0]])

        def apply_flocking_rules(self, neighbors, separation_dist=10, cohesion_weight=0.01, alignment_weight=0.1, separation_weight=0.1):
            if len(neighbors) == 0:
                return

            center_of_mass = np.mean([boid.position for boid in neighbors], axis=0)
            avg_velocity = np.mean([boid.velocity for boid in neighbors], axis=0)
            separation_force = np.sum([self.position - boid.position for boid in neighbors if np.linalg.norm(self.position - boid.position) < separation_dist], axis=0)

            if len(neighbors) > 0:
                separation_force /= len(neighbors)

            cohesion_force = (center_of_mass - self.position) * cohesion_weight
            alignment_force = (avg_velocity - self.velocity) * alignment_weight
            self.velocity += cohesion_force + alignment_force + separation_force * separation_weight

        def attract_to(self, target, weight=0.05):
            direction = np.array(target, dtype=float) - self.position
            self.velocity += direction * weight

        def get_color_based_on_velocity(self):
            hue = (np.arctan2(self.velocity[1], self.velocity[0]) + np.pi) / (2 * np.pi)
            speed = np.linalg.norm(self.velocity)
            brightness = min(speed / 5.0, 1.0)
            rgb = np.array(colorsys.hsv_to_rgb(hue, 1.0, brightness))
            return (rgb * 255).astype(np.uint8)

    def update_frame_and_boids(new_frame, boids):
        frame_shape = new_frame.shape
        frame_height, frame_width = frame_shape[:2]

        # Rescale the frame to half size for performance optimization
        small_frame = cv2.resize(new_frame, (frame_width // 2, frame_height // 2))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        thresh_frame = cv2.adaptiveThreshold(
            blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        frame_height, frame_width = small_frame.shape[:2]

        attraction_points = np.argwhere(thresh_frame == 255)
        if attraction_points.size > 0:
            tree = KDTree(attraction_points)

            positions = np.array([boid.position for boid in boids])
            boid_tree = KDTree(positions)

            for boid in boids:
                dist, idx = tree.query(boid.position[::-1], k=1)
                closest_point = attraction_points[idx]
                boid.attract_to(closest_point[::-1])

                neighbor_idxs = boid_tree.query_ball_point(boid.position, r=25)  # Adjust for resized frame
                neighbors = [boids[i] for i in neighbor_idxs if i != boid]

                boid.apply_flocking_rules(neighbors)

                boid.move(frame_shape=(frame_height, frame_width))

                # Reduce randomness to limit unnecessary changes
                if np.random.rand() < 0.001:
                    boid.position = np.random.rand(2) * np.array([frame_width, frame_height])

        painted_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        painted_frame = np.clip(painted_frame * 0.98, 0, 255).astype(np.uint8)

        for boid in boids:
            color = boid.get_color_based_on_velocity()
            cv2.circle(painted_frame, tuple(boid.position.astype(int)), boid.radius, color.tolist(), -1)

        return thresh_frame, painted_frame

    @thread_worker(connect={"yielded": lambda data: update_viewer(data)})
    def frame_generator(boids):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = frame[:, ::-1]
            yield update_frame_and_boids(frame, boids)
            time.sleep(1 / 30)

        cap.release()

    @ensure_main_thread
    def update_viewer(frame_data):
        webcam_frame, boid_paint_frame = frame_data
        viewer.layers["Webcam"].data = cv2.cvtColor(webcam_frame, cv2.COLOR_GRAY2BGR)
        viewer.layers["Boid Paint"].data = boid_paint_frame

    viewer = napari.Viewer()
    viewer.window.resize(800, 600)

    # Adjusted for half-size frame
    boid_paint_layer = viewer.add_image(np.zeros((240, 320, 3), dtype=np.uint8), name="Boid Paint", opacity=1.0)
    webcam_layer = viewer.add_image(np.zeros((240, 320, 3), dtype=np.uint8), name="Webcam", opacity=0.5)

    frame_shape = (240, 320, 3)

    num_boids = 100
    boids = [Boid(position=np.random.rand(2) * np.array([frame_shape[1], frame_shape[0]]), velocity=np.random.rand(2) * 2 - 1) for _ in range(num_boids)]

    worker = frame_generator(boids)

    napari.run()


setup(
    group="collective-dynamics",
    name="boids-webcam",
    version="0.0.1",
    title="Webcam with Boids",
    description="An Album solution that captures a webcam feed and simulates boids interacting with the feed using napari.",
    solution_creators=["Kyle Harrington"],
    cite=[{
        "text": "Reynolds, C.W., 1987. Flocks, herds and schools: A distributed behavioral model. ACM SIGGRAPH Computer Graphics, 21(4), pp.25-34."
    }],
    tags=["Boids", "flocking", "napari", "webcam"],
    license="MIT",
    covers=[{
        "description": "Webcam with Boids cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
    args=[],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "interactive-napari",
            "version": "0.0.1"
        },
    }
)
