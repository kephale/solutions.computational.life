###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import cv2
    import napari
    import numpy as np
    import pyvirtualcam
    from magicgui import magic_factory
    from napari.qt.threading import thread_worker

    import time
    import logging

    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    LOGGER = logging.getLogger(__name__)

    # Define a class to hold the state
    class WebcamState:
        def __init__(self, radius=5):
            self.capturing = False
            self.trail_param = 0.1
            self.ant_position = None
            self.ant_direction = 0  # 0=up, 1=right, 2=down, 3=left
            self.ant_color = (0, 0, 255)  # Red color for the ant in BGR format (tuple)
            self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            self.grid = None  # To be initialized based on webcam frame size
            self.radius = radius
            self.step_size = radius  # Set the step size based on the radius

        def initialize_grid(self, shape):
            # Initialize the grid based on the webcam frame size
            self.grid = np.zeros((shape[0], shape[1]), dtype=np.uint8)
            self.ant_position = [shape[0] // 2, shape[1] // 2]  # Center the ant

        def update_ant(self, input_grid):
            # Flip the cell state based on the ant's radius
            current_state = input_grid[self.ant_position[0], self.ant_position[1]]
            new_state = 255 if current_state == 0 else 0  # Ensuring the state is a scalar
            cv2.circle(self.grid, tuple(self.ant_position[::-1]), self.radius, new_state, -1)

            # Turn the ant: right on white (0), left on black (1)
            if current_state == 0:
                self.ant_direction = (self.ant_direction + 1) % 4
            else:
                self.ant_direction = (self.ant_direction - 1) % 4

            # Move the ant forward with step size proportional to the radius
            self.ant_position[0] = (self.ant_position[0] + self.directions[self.ant_direction][0] * self.step_size) % self.grid.shape[0]
            self.ant_position[1] = (self.ant_position[1] + self.directions[self.ant_direction][1] * self.step_size) % self.grid.shape[1]

    # Create an instance of the state with the specified ant radius
    global state
    state = WebcamState(radius=5)

    def make_layer(layer_name="Conference", viewer=None):
        global state
        LOGGER.info("Entering make_layer function")

        def update_layer(new_frame):
            global state
            if state.grid is None:
                state.initialize_grid(new_frame.shape[:2])

            try:
                viewer.layers[layer_name].data = new_frame
            except KeyError:
                LOGGER.info("Adding new layer")
                viewer.add_image(new_frame, name=layer_name)

            # Update Langton's Ant
            state.update_ant(new_frame[:,:,0])

            # Draw the ant in red on the webcam feed
            ant_frame = new_frame.copy()
            cv2.circle(ant_frame, tuple(state.ant_position[::-1]), state.radius, state.ant_color, -1)

            # Adjust the brightness of the trail to ensure it stands out
            trail_image = np.stack([state.grid] * 3, axis=-1)  # Trail matches the radius
            trail_image[state.grid == 255] = [0, 50, 255]  # Dark gray for the trail

            # Combine the webcam feed and Langton's Ant grid
            combined_frame = cv2.addWeighted(ant_frame, 1.0, trail_image, 0.5, 0)

            viewer.layers[layer_name].data = combined_frame

            screen = viewer.screenshot(flash=False, canvas_only=False)

        @thread_worker
        def frame_updater():
            global state
            LOGGER.info("Opening webcam")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                LOGGER.error("Cannot open webcam")
                raise IOError("Cannot open webcam")

            prev_frame = None
            LOGGER.info("Starting frame capturing loop")
            while state.capturing:
                ret, frame = cap.read()
                if not ret:
                    LOGGER.error("Failed to read frame from webcam")
                    continue

                # Resize and convert to grayscale
                frame = cv2.resize(
                    frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                )
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

                # Adaptive thresholding to highlight the face
                thresh_frame = cv2.adaptiveThreshold(
                    blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )

                # Convert the single-channel thresholded image to a 3-channel image
                frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

                if prev_frame is None:
                    prev_frame = frame

                frame = np.array(
                    prev_frame * state.trail_param
                    + frame * (1.0 - state.trail_param),
                    dtype=frame.dtype,
                )
                prev_frame = np.array(frame)
                yield frame

                # Sleep for 20FPS
                time.sleep(1 / 20)
            cap.release()
            LOGGER.info("Capture device released.")

        LOGGER.info("Starting frame updater worker")
        worker = frame_updater()
        worker.yielded.connect(update_layer)
        worker.start()

        return worker

    @magic_factory(
        call_button="Update",
    )
    def conference_widget(
        viewer: "napari.viewer.Viewer",
        layer_name="Napari Conference",
        running=False,
        trails_param=0.1,
        radius=5,  # Add radius as a parameter
    ):
        global state

        state.capturing = running
        state.trail_param = trails_param
        state.radius = radius  # Update the radius in the state
        state.step_size = radius  # Update the step size based on the radius

        if state.capturing:
            LOGGER.info("Creating layer")
            make_layer(layer_name, viewer=viewer)

    viewer = napari.Viewer()
    viewer.window.resize(800, 600)

    widget = conference_widget()

    viewer.window.add_dock_widget(widget, name="Conference")

    napari.run()

setup(
    group="interactive",
    name="langtons-ant-webcam",
    version="0.0.1",
    title="Webcam with Langton's Ant",
    description="An Album solution that captures a webcam feed and simulates Langton's Ant interacting with the feed using napari.",
    authors=["Kyle Harrington"],
    cite=[{
        "text": "Langton, C.G., 1986. Studying artificial life with cellular automata. Physica D: nonlinear phenomena, 22(1-3), pp.120-149."
    }],
    tags=["Langton's Ant", "automata", "napari", "webcam"],
    license="MIT",
    covers=[{
        "description": "Webcam with Langton's Ant cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
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
