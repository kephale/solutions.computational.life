###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import numpy as np
    import napari
    import random
    from skimage import draw
    from vispy.color import Color
    from qtpy.QtCore import QTimer
    from noise import pnoise2

    class Particle:
        def __init__(self, x, y, colors):
            self.x = x
            self.y = y
            self.pos = np.array([self.x, self.y], dtype=np.float32)
            self.life = np.random.rand()
            self.c = random.choice(colors)
            self.ff = 0

        def update(self):
            scale = 0.01
            self.ff = pnoise2(self.pos[0] * scale, self.pos[1] * scale) * 2 * np.pi
            mainP = 1200
            changeDir = 2 * np.pi / mainP
            roundff = round((self.ff / (2 * np.pi)) * mainP)
            self.ff = changeDir * roundff

            if 3 < self.ff < 6:
                self.c = colors[0]
                self.pos += np.array([np.tan(self.ff) * np.random.uniform(1, 3), np.tan(self.ff)])
            else:
                self.c = colors[1]
                self.pos -= np.array([np.sin(self.ff) * np.random.uniform(0.1, 1), np.cos(self.ff)])

        def show(self, image):
            lx, ly = 20, 20
            px = np.clip(self.pos[0], lx, image.shape[1] - lx)
            py = np.clip(self.pos[1], ly, image.shape[0] - ly)
            rr, cc = draw.disk((py, px), radius=np.random.uniform(1, 1.25))
            image[rr, cc] = self.c[:3] * 255

        def finished(self):
            self.life -= np.random.rand() ** 4 / 10
            self.life = np.clip(self.life, 0, 1)
            return self.life == 0

    def create_particles(num_particles, width, height, colors):
        return [Particle(random.uniform(0, width), random.uniform(0, height), colors) for _ in range(num_particles)]

    colors = [
        np.array([15 / 360, 0.9, 0.9]),
        np.array([175 / 360, 0.9, 0.9])
    ]

    parNum = 1000
    image_size = 512

    particles = create_particles(parNum, image_size, image_size, colors)
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    viewer = napari.Viewer()
    layer = viewer.add_image(image, rgb=True)

    def update_frame():
        fade_factor = 0.9
        image[:] = (image[:] * fade_factor).astype(np.uint8)
        for particle in particles[:]:
            particle.update()
            particle.show(image)
            if particle.finished():
                particles.remove(particle)

        while len(particles) < parNum:
            particles.append(Particle(random.uniform(0, image.shape[1]), random.uniform(0, image.shape[0]), colors))

        layer.data = image

    # Use a QTimer to periodically call the update function
    timer = QTimer()
    timer.timeout.connect(update_frame)
    timer.start(100)

    napari.run()

setup(
    group="patterns",
    name="particle-sands",
    version="0.0.1",
    title="Particle sands 02 pattern",
    description="An Album solution that generates particle patterns and displays them using Napari.",
    authors=["Kyle Harrington"],
    cite=[{"text": "Original by Samuel Yan: https://openprocessing.org/user/293890?view=sketches&o=48.", "url": "https://openprocessing.org/sketch/1353598"}],
    tags=["particles", "pattern", "python", "napari", "particles"],
    license="MIT",
    covers=[{
        "description": "Particle Pattern Generation cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "napari",
            "version": "0.0.3"
        }
    }
)
