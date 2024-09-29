###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import napari
    import torch
    import numpy as np
    from superqt import ensure_main_thread
    from superqt.utils import thread_worker
    import time
    from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
    from qtpy.QtCore import Qt

    # Autodetect the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Parameters for the BZ reaction
    alpha, beta, gamma = 1.0, 1.0, 1.0

    # Class to manage the shared pause state and worker control
    class SimulationState:
        def __init__(self):
            self.paused = False
            self.worker = None

        def toggle_pause(self):
            self.paused = not self.paused

        def is_paused(self):
            return self.paused

        def stop_worker(self):
            if self.worker is not None:
                self.worker.quit()  # Safely stop the worker
                self.worker = None

        def set_worker(self, worker):
            self.stop_worker()  # Ensure previous worker is stopped
            self.worker = worker
            self.worker.start()  # Start the new worker

    # Initialize the chemical concentrations with random values
    def initialize_reaction(size, device):
        arr = torch.rand((2, 3, size[0], size[1]), device=device)
        return arr

    # Update function for the reaction-diffusion system
    def update_reaction(p, arr, alpha, beta, gamma):
        q = (p + 1) % 2
        s = torch.zeros((3, arr.shape[2], arr.shape[3]), device=device)
        m = torch.ones((3, 3), device=device) / 9

        # Perform convolution for the 9-cell neighborhood average
        for k in range(3):
            s[k] = torch.nn.functional.conv2d(arr[p, k][None, None, :, :], m[None, None, :, :], padding='same')[0, 0, :, :]

        # Apply the reaction equations (simplified Oregonator model)
        arr[q, 0] = s[0] + s[0] * (alpha * s[1] - gamma * s[2])
        arr[q, 1] = s[1] + s[1] * (beta * s[2] - alpha * s[0])
        arr[q, 2] = s[2] + s[2] * (gamma * s[0] - beta * s[1])

        # Ensure the species concentrations are kept within [0, 1]
        arr[q] = torch.clamp(arr[q], 0, 1)
        
        return arr, q

    # Generator function that yields frames
    @thread_worker
    def frame_generator(arr, alpha, beta, gamma, state):
        p = 0
        while True:
            if not state.is_paused():
                arr, p = update_reaction(p, arr, alpha, beta, gamma)
                yield arr[p].cpu().numpy()  # Yielding the CPU version for UI compatibility
            time.sleep(0.02)

    # Function to restart the worker
    def start_worker(arr, alpha, beta, gamma, state, update_layers):
        worker = frame_generator(arr, alpha, beta, gamma, state)
        worker.yielded.connect(update_layers)
        state.set_worker(worker)  # Stop the old worker and start a new one

    # Example usage
    size = (450, 600)  # Size of the grid (height, width)
    arr = initialize_reaction(size, device)

    # Create the Napari viewer
    viewer = napari.Viewer()

    # Add empty image layers for each chemical concentration
    u_layer = viewer.add_image(np.zeros(size), name="Species A (U)")
    v_layer = viewer.add_image(np.zeros(size), name="Species B (V)")
    w_layer = viewer.add_image(np.zeros(size), name="Species C (W)")

    # Function to update the image layers using ensure_main_thread
    @ensure_main_thread
    def update_layers(frame):
        u_layer.data = frame[0]
        v_layer.data = frame[1]
        w_layer.data = frame[2]

    # Create control widget
    class ControlWidget(QWidget):
        def __init__(self, state):
            super().__init__()
            self.state = state
            self.initUI()

        def initUI(self):
            layout = QVBoxLayout()

            # Pause/Resume button
            self.pause_button = QPushButton("Pause")
            self.pause_button.clicked.connect(self.toggle_pause)
            layout.addWidget(self.pause_button)

            # Reinitialize button
            self.reinit_button = QPushButton("Reinitialize")
            self.reinit_button.clicked.connect(self.reinitialize)
            layout.addWidget(self.reinit_button)

            # Sliders for controlling parameters
            self.alpha_slider = self.create_slider("Alpha", 1.0, 0.0, 2.0, layout)
            self.beta_slider = self.create_slider("Beta", 1.0, 0.0, 2.0, layout)
            self.gamma_slider = self.create_slider("Gamma", 1.0, 0.0, 2.0, layout)

            self.setLayout(layout)

        def create_slider(self, name, initial_value, min_value, max_value, layout):
            label = QLabel(f"{name}: {initial_value}")
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_value * 100))
            slider.setMaximum(int(max_value * 100))
            slider.setValue(int(initial_value * 100))
            slider.valueChanged.connect(lambda value: self.update_value(label, name, value / 100))
            layout.addWidget(label)
            layout.addWidget(slider)
            return slider

        def update_value(self, label, name, value):
            label.setText(f"{name}: {value}")
            nonlocal alpha, beta, gamma
            if name == "Alpha":
                alpha = value
            elif name == "Beta":
                beta = value
            elif name == "Gamma":
                gamma = value
            # Restart the worker to apply new parameter values
            start_worker(arr, alpha, beta, gamma, self.state, update_layers)

        def toggle_pause(self):
            self.state.toggle_pause()
            if self.state.is_paused():
                self.pause_button.setText("Resume")
            else:
                self.pause_button.setText("Pause")

        def reinitialize(self):
            nonlocal arr
            arr = initialize_reaction(size, device)
            start_worker(arr, alpha, beta, gamma, self.state, update_layers)  # Restart the worker with new array

    # Shared simulation state
    state = SimulationState()

    # Add the control widget to the Napari viewer
    control_widget = ControlWidget(state)
    viewer.window.add_dock_widget(control_widget, name="Controls", area="right")

    # Start the frame generation worker initially
    start_worker(arr, alpha, beta, gamma, state, update_layers)

    # Start the Napari event loop
    napari.run()

setup(
    group="reaction-diffusion",
    name="belousov-zhabotinsky",
    version="0.0.3",
    title="Belousov-Zhabotinsky Reaction Simulation",
    description="Simulates the Belousov-Zhabotinsky reaction using the Oregonator model.",
    solution_creators=["Kyle Harrington"],
    cite=[{
        "text": "Field, R.J., Körös, E., Noyes, R.M., 1972. Oscillations in chemical systems. I. Detailed mechanism in a system showing temporal oscillations. Journal of the American Chemical Society, 94(25), pp.8649-8664."
    }],
    tags=["reaction-diffusion", "BZ reaction", "automata", "napari", "simulation"],
    license="MIT",
    covers=[{
        "description": "Belousov-Zhabotinsky Reaction Simulation cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.5.1",
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
