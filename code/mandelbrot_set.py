import os.path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def mandelbrot(c, max_iter):
    """Compute the number of iterations for the Mandelbrot set."""
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_image(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Generate a Mandelbrot set image."""
    image = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            re = xmin + (x / width) * (xmax - xmin)
            im = ymin + (y / height) * (ymax - ymin)
            c = complex(re, im)
            image[y, x] = mandelbrot(c, max_iter)
    return image

center_x = -0.7436438870371587
center_y =  0.13182590420533

IMG_PATH = os.path.join(os.path.dirname(__file__), "../img")

if __name__ == "__main__":
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 250)

    frames = []
    for zoom in range(30):
        scale = 2 ** (-0.5 * zoom)
        width_range = 3.5 * scale
        height_range = 2.5 * scale

        xmin = center_x - width_range / 2
        xmax = center_x + width_range / 2
        ymin = center_y - height_range / 2
        ymax = center_y + height_range / 2

        filename = f"{IMG_PATH}/mandelbrot_{zoom:03d}.png"
        img = generate_image(xmin, xmax, ymin, ymax, 400, 400, 100)
        plt.imsave(filename, img, cmap='inferno')
        frames.append(Image.open(filename))

    frames[0].save(f'{IMG_PATH}/mandelbrot_zoom.gif',
                   save_all=True,
                   append_images=frames[1:],
                   duration=100,
                   loop=0)
