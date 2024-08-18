###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import napari
    import torch
    import numpy as np
    from qtpy.QtCore import QTimer

    # Autodetect the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Parameters for the BZ reaction
    alpha, beta, gamma = 1.0, 1.0, 1.0

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
    def frame_generator(arr, alpha, beta, gamma):
        p = 0
        while True:
            arr, p = update_reaction(p, arr, alpha, beta, gamma)
            yield arr[p]

    # Example usage
    size = (450, 600)  # Size of the grid (height, width)
    arr = initialize_reaction(size, device)
    gen = frame_generator(arr, alpha, beta, gamma)

    # Create the Napari viewer
    viewer = napari.Viewer()

    # Add empty image layers for each chemical concentration
    u_layer = viewer.add_image(np.zeros(size), name="Species A (U)")
    v_layer = viewer.add_image(np.zeros(size), name="Species B (V)")
    w_layer = viewer.add_image(np.zeros(size), name="Species C (W)")

    # Function to update the image layers
    def update_frame():
        try:
            # Get the next frames from the generator
            frame = next(gen)
            # Update the image layers with the new frames
            u_layer.data = frame[0].cpu().numpy()
            v_layer.data = frame[1].cpu().numpy()
            w_layer.data = frame[2].cpu().numpy()
        except StopIteration:
            # Stop the timer when the generator is exhausted
            timer.stop()

    # Create a QTimer to periodically update the image layers
    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(100)  # Update every 100 milliseconds

    # Start the Napari event loop
    napari.run()

setup(
    group="reaction-diffusion",
    name="belousov-zhabotinsky",
    version="0.0.1",
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
