###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import cv2
    import mediapipe as mp
    import numpy as np
    import napari
    from qtpy.QtCore import Signal, QObject
    from napari.qt.threading import thread_worker
    import time
    from magicgui import magicgui

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    class PoseLandmarks(QObject):
        updated = Signal()

        def __init__(self):
            super().__init__()
            self.landmarks = None

        def process_image(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks = pose.process(image_rgb)
            if landmarks.pose_landmarks:
                self.landmarks = landmarks.pose_landmarks.landmark
                self.updated.emit()

    pose_landmarks = PoseLandmarks()

    # Boid class to model individual boid behaviors
    class Boid:
        def __init__(self, position, velocity, alignment_weight=0.2, cohesion_weight=0.5, separation_weight=1, attract_to_hands_weight=1, attract_to_origin_weight=0.0, box_bounds=np.array([1000, 1000, 1000])):
            self.position = np.array(position, dtype=np.float64)
            self.velocity = np.array(velocity, dtype=np.float64)
            self.acceleration = np.array([0, 0, 0], dtype=np.float64)
            self.speed = 20
            self.max_force = 20.0  # Increased max force
            self.max_velocity = 10.0  # Cap the velocity
            self.perception = 500
            self.alignment_weight = alignment_weight
            self.cohesion_weight = cohesion_weight
            self.separation_weight = separation_weight
            self.attract_to_hands_weight = attract_to_hands_weight
            self.attract_to_origin_weight = attract_to_origin_weight
            self.box_bounds = box_bounds

        def apply_force(self, force):
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > self.max_force:
                force = (force / force_magnitude) * self.max_force
            self.acceleration += force

        def normalize_force(self, force):
            force_magnitude = np.linalg.norm(force)
            if force_magnitude > 0:
                return force / force_magnitude
            return force

        def alignment(self, boids):
            steering = np.zeros(3)
            total = 0
            avg_velocity = np.zeros(3, dtype=np.float64)

            for boid in boids:
                if np.linalg.norm(boid.position - self.position) < self.perception:
                    avg_velocity += boid.velocity
                    total += 1

            if total > 0:
                avg_velocity /= total
                steering = avg_velocity - self.velocity
            return steering

        def cohesion(self, boids):
            steering = np.zeros(3)
            total = 0
            center_of_mass = np.zeros(3, dtype=np.float64)

            for boid in boids:
                if np.linalg.norm(boid.position - self.position) < self.perception:
                    center_of_mass += boid.position
                    total += 1

            if total > 0:
                center_of_mass /= total
                vector_to_com = center_of_mass - self.position
                if np.linalg.norm(vector_to_com) > 0:
                    steering = vector_to_com - self.velocity
            return steering

        def separation(self, boids):
            steering = np.zeros(3)
            total = 0

            for boid in boids:
                distance = np.linalg.norm(boid.position - self.position)
                if 0 < distance < self.perception:
                    diff = self.position - boid.position
                    diff /= (distance ** 2)
                    steering += diff
                    total += 1

            if total > 0:
                steering /= total
            return steering

        def attract_to_hands(self, hand_position):
            vector_to_hand = hand_position - self.position
            distance_to_hand = np.linalg.norm(vector_to_hand)
            if distance_to_hand > 0:
                desired_velocity = (vector_to_hand / distance_to_hand) * self.speed
                steering = desired_velocity - self.velocity
                return steering
            return np.zeros(3)

        def bound_position(self):
            self.position = np.clip(self.position, -1 * np.array(self.box_bounds), self.box_bounds)

        def limit_velocity(self):
            velocity_magnitude = np.linalg.norm(self.velocity)
            if velocity_magnitude > self.max_velocity:
                self.velocity = (self.velocity / velocity_magnitude) * self.max_velocity

        def update(self):
            self.velocity += self.acceleration
            self.limit_velocity()
            self.position += self.velocity
            self.bound_position()
            self.acceleration = np.zeros(3, dtype=np.float64)

    # Thread worker for webcam feed
    @thread_worker(connect={"yielded": lambda frame: update_points_and_webcam(frame)})
    def worker(video_path=0):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pose_landmarks.process_image(frame)
            yield frame
            time.sleep(0.01)

    viewer = napari.Viewer(ndisplay=3)

    # 3D image layer for painting boid positions
    image_layer = viewer.add_image(np.zeros((100, 100, 100)), name="Boid Paint", colormap="magma", blending="additive", scale=(10, 10, 10), translate=(0, 0, -1000))
    image_layer.bounding_box.visible = True

    # Boids initialization
    boids = [Boid(np.random.rand(3) * 1000, np.random.rand(3) * 25, box_bounds=np.array([1000, 1000, 1000])) for _ in range(25)]
    boids_layer = viewer.add_points(np.array([boid.position for boid in boids]), face_color='blue', size=50)

    # Add webcam layer
    points_layer = viewer.add_points(np.zeros((33, 3)), face_color='red', size=100)
    webcam_layer = viewer.add_image(np.zeros((240, 320, 1)), name="Webcam", colormap="gray", opacity=0.5)

    x_position, y_position, z_position = 0, 0, 0
    webcam_layer.translate = (x_position, y_position, z_position)

    @magicgui(alignment_weight={"label": "Alignment"}, cohesion_weight={"label": "Cohesion"}, separation_weight={"label": "Separation"}, attract_to_hands_weight={"label": "Attract to Hands"})
    def adjust_boid_weights(alignment_weight=0.2, cohesion_weight=0.5, separation_weight=1.0, attract_to_hands_weight=1.0):
        for boid in boids:
            boid.alignment_weight = alignment_weight
            boid.cohesion_weight = cohesion_weight
            boid.separation_weight = separation_weight
            boid.attract_to_hands_weight = attract_to_hands_weight

    viewer.window.add_dock_widget(adjust_boid_weights, area='right')

    def update_points_and_webcam(frame):
        if pose_landmarks.landmarks:
            coords = [(lm.x * 1000, lm.y * 1000, lm.z * 1000) for lm in pose_landmarks.landmarks]
            colors = [(1, 0, 0, 1) if lm.visibility > 0.5 else (0.5, 0.5, 0.5, 0.5) for lm in pose_landmarks.landmarks]
            points_layer.data = coords
            points_layer.face_color = colors

            if frame is not None:
                small_frame = cv2.resize(frame, (320, 240))
                gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                transposed_frame = cv2.transpose(gray_frame)
                transposed_frame = cv2.equalizeHist(transposed_frame)
                edges = cv2.Canny(transposed_frame, threshold1=50, threshold2=150)
                webcam_layer.data = np.expand_dims(edges, axis=-1)
                webcam_layer.scale = (4, 4, 1)

            left_wrist = np.array([pose_landmarks.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * 1000,
                                   pose_landmarks.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * 1000,
                                   pose_landmarks.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z * 1000])
            right_wrist = np.array([pose_landmarks.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * 1000,
                                    pose_landmarks.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * 1000,
                                    pose_landmarks.landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z * 1000])

            for boid in boids:
                alignment_force = boid.normalize_force(boid.alignment(boids)) * boid.alignment_weight
                cohesion_force = boid.normalize_force(boid.cohesion(boids)) * boid.cohesion_weight
                separation_force = boid.normalize_force(boid.separation(boids)) * boid.separation_weight
                attract_left = boid.normalize_force(boid.attract_to_hands(left_wrist)) * boid.attract_to_hands_weight
                attract_right = boid.normalize_force(boid.attract_to_hands(right_wrist)) * boid.attract_to_hands_weight

                total_force = alignment_force + cohesion_force + separation_force + attract_left + attract_right
                boid.apply_force(total_force)
                boid.update()

            # Update boid positions in 3D image
            # image_layer.data.fill(0)
            image_layer.data *= 0.999
            for boid in boids:
                x, y, z = ((boid.position - image_layer.translate) / image_layer.scale).astype(int)
                # print(f"painting {boid.position} and {(x, y, z)}")
                if 0 <= x < 100 and 0 <= y < 100 and 0 <= z < 100:
                    image_layer.data[x, y, z] += 1

            # image_layer.contrast_limits = (0, image_layer.contrast_limits[1] + 1)
            image_layer.contrast_limits = (0, np.max(image_layer.data) + 1)
            boids_layer.data = np.array([boid.position for boid in boids])
        # Camera(center=(157.02459430119973, 178.03531960150553, -328.39814014159066), zoom=0.38332499999999997, angles=(66.26533421436382, -80.61184360683399, 25.264526269075716), perspective=0.0, mouse_pan=True, mouse_zoom=True)

    # Camera(center=(744.1635097100659, 561.8353434705525, 454.6789509125847), zoom=0.71756877609384, angles=(19.294056138962944, -79.74868537277544, 70.01867541676341), perspective=0.0, mouse_pan=True, mouse_zoom=True)

    viewer.camera.zoom = 0.71756877609384
    viewer.camera.angles = (19.294056138962944, -79.74868537277544, 70.01867541676341)
    viewer.camera.center = (744.1635097100659, 561.8353434705525, 454.6789509125847)

    worker()
    napari.run()

setup(
    group="collective-dynamics",
    name="boids-paint",
    version="0.0.2",
    title="3D Boids Paint with Pose Tracking controlling boids",
    description="An Album solution that simulates boids interacting with hand landmarks tracked from webcam and paints boid positions into a 3D image.",
    solution_creators=["Kyle Harrington and Iris Harrington"],
    cite=[{
        "text": "Reynolds, C.W., 1987. Flocks, herds and schools: A distributed behavioral model. ACM SIGGRAPH Computer Graphics, 21(4), pp.25-34."
    }],
    tags=["Boids", "flocking", "pose", "3D", "napari", "webcam"],
    covers=[
        {
            "description": "Painting a 3D volume using a swarm controlled by body.",
            "source": "cover.png",
        }
    ],    
    license="MIT",
    album_api_version="0.5.1",    
    args=[],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "interactive-napari",
            "version": "0.0.2"
        },
    }
)
