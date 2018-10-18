from __future__ import division
import numpy as np
from utils import unit_vector


def render_image(nx, ny, ns, scene, camera):
    """
    Render an image via raytracing.

    Args:
        nx: width of rendered image (pixels)
        ny: height of rendered image (pixels)
        ns: number of samples (rays) per pixel
        scene: geometric scene
        camera: camera for generating rays

    Returns:
        numpy.ndarray: rendered image of shape `(nx, ny, 3)`
    """
    # fill image pixels
    im = np.zeros((nx, ny, 3), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            col = np.zeros(3)
            for s in range(ns):
                # add a random offset for antialiasing
                u = (i + np.random.rand()) / nx
                v = (j + np.random.rand()) / ny
                ray = camera.get_ray(u, v)
                col += ray_color(ray, scene, 50)
            col /= ns

            # take sqrt for gamma correction
            im[i, -(j + 1)] = np.round(255 * np.sqrt(col)).astype(int)

    return im


def ray_color(ray, scene, depth):
    """
    Perform ray tracing and return color of ray.

    Args:
        ray: to-be traced ray
        scene: geometric scene
        depth: how often the ray is allowed to scatter

    Returns:
        numpy.ndarray: ray color as RGB values
    """
    rec, _ = scene.hit(ray, 0.001, 1e6)
    if rec is not None:
        scattered, attenuation = rec.material.scatter(ray, rec)
        if depth > 0 and scattered is not None:
            # pointwise multiplication between attenuation and
            # return value of recursive function call
            return attenuation * ray_color(scattered, scene, depth - 1)
        else:
            return np.zeros(3)
    else:
        # blue background sky
        unitdir = unit_vector(ray.direction)
        t = 0.5*(unitdir[1] + 1)
        return (1 - t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])
