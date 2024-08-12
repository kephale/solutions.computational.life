###album catalog: solutions.computational.life

from album.runner.api import get_args, setup

def run():
    import numpy as np
    import napari
    from skimage.draw import line

    # Function to apply L-system rules
    def apply_lsystem_rules(axiom, rules, iterations):
        for _ in range(iterations):
            axiom = ''.join([rules.get(char, char) for char in axiom])
        return axiom

    # Function to generate the Barnsley Fern L-system
    def barnsley_fern_lsystem(axiom='X', iterations=5):
        rules = {
            'X': 'F+[[X]-X]-F[-FX]+X',
            'F': 'FF',
            '+': '+',
            '-': '-',
            '[': '[',
            ']': ']',
        }
        return apply_lsystem_rules(axiom, rules, iterations)

    # Function to interpret the L-system and generate points
    def draw_lsystem(lsystem_string, angle=25):
        position_stack = []
        positions = []
        color_gradients = []
        angle_stack = []
        current_position = np.array([0, 0], dtype=float)
        current_angle = -90  # Start upside down

        color_start = np.array([139, 69, 19], dtype=float)  # SaddleBrown
        color_end = np.array([34, 139, 34], dtype=float)    # ForestGreen
        color_delta = (color_end - color_start) / len(lsystem_string)

        for i, char in enumerate(lsystem_string):
            if char == 'F':
                new_position = current_position + np.array([
                    np.cos(np.radians(current_angle)),
                    np.sin(np.radians(current_angle))
                ])
                positions.append((current_position, new_position))
                color_gradients.append(color_start + i * color_delta)
                current_position = new_position
            elif char == '+':
                current_angle += angle
            elif char == '-':
                current_angle -= angle
            elif char == '[':
                position_stack.append(current_position.copy())
                angle_stack.append(current_angle)
            elif char == ']':
                current_position = position_stack.pop()
                current_angle = angle_stack.pop()
        
        return positions, color_gradients

    # Function to calculate the bounding box of the L-system
    def calculate_bounding_box(positions):
        min_x = min(min(start[0], end[0]) for start, end in positions)
        max_x = max(max(start[0], end[0]) for start, end in positions)
        min_y = min(min(start[1], end[1]) for start, end in positions)
        max_y = max(max(start[1], end[1]) for start, end in positions)
        return min_x, max_x, min_y, max_y

    # Function to render the L-system in napari
    def render_lsystem_in_napari(positions, color_gradients):
        min_x, max_x, min_y, max_y = calculate_bounding_box(positions)
        canvas_width = int(max_x - min_x + 1)
        canvas_height = int(max_y - min_y + 1)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # RGB canvas

        for (start, end), color in zip(positions, color_gradients):
            rr, cc = line(int(start[1] - min_y), int(start[0] - min_x),
                          int(end[1] - min_y), int(end[0] - min_x))
            canvas[rr, cc, :] = color.astype(np.uint8)
        
        viewer = napari.Viewer()
        viewer.add_image(canvas, name="Barnsley Fern L-system")
        napari.run()

    # Get arguments from the user (optional)
    args = get_args()
    angle = args.angle
    iterations = args.iterations

    # Generate the L-system string for Barnsley Fern
    lsystem_string = barnsley_fern_lsystem(axiom='X', iterations=iterations)

    # Interpret and draw the L-system
    positions, color_gradients = draw_lsystem(lsystem_string, angle=angle)

    # Render the L-system in Napari
    render_lsystem_in_napari(positions, color_gradients)

setup(
    group="l-systems",
    name="barnsley-fern",
    version="0.0.5",
    title="Barnsley Fern L-System",
    description="An album solution that generates and displays a Barnsley Fern L-system using napari. The fern transitions from brown to green as it grows.",
    authors=["Kyle Harrington"],
    cite=[{
        "text": "Prusinkiewicz, P., & Lindenmayer, A. (1990). The algorithmic beauty of plants. Springer Science & Business Media."
    }],
    tags=["L-System", "Barnsley Fern", "generative", "napari", "simulation"],
    license="MIT",
    covers=[{
        "description": "Barnsley Fern L-System cover image.",
        "source": "cover.png"
    }],
    album_api_version="0.3.1",
    args=[
        {
            "name": "angle",
            "type": "float",
            "description": "The angle in degrees for each turn.",
            "required": False,
            "default": 25
        },
        {
            "name": "iterations",
            "type": "integer",
            "description": "The number of iterations to generate the L-system string.",
            "required": False,
            "default": 5
        }
    ],
    run=run,
    dependencies={
        "parent": {
            "group": "environments",
            "name": "napari",
            "version": "0.0.7"
        }
    }
)
