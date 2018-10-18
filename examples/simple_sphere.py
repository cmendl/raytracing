from __future__ import division
import numpy as np
import imageio
import sys
sys.path.append('../engine/')
from surface import SurfaceAssembly, Sphere
from material import Lambertian
from camera import Camera
from rendering import ray_color


def main():

    # image dimensions (pixel)
    nx = 200
    ny = 100

    # number of samples for anti-aliasing
    ns = 100

    # define camera
    lookfrom = np.zeros(3)
    lookat = np.array([0., 0., -1.])
    vfov = np.pi/2
    aperture = 0.
    focus_dist = 1.
    cam = Camera(lookfrom, lookat, np.array([0., 1., 0.]), vfov, nx / ny, aperture, focus_dist)

    mat = Lambertian(0.5)

    # define world geometry
    world = SurfaceAssembly()
    world.add_object(Sphere(np.array([0.,    0.,  -1.]), 0.5, mat))
    world.add_object(Sphere(np.array([0., -100.5, -1.]), 100, mat))

    # fill image pixels
    im = np.zeros((nx, ny, 3), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            col = np.zeros(3)
            for s in range(ns):
                u = (i + np.random.rand()) / nx
                v = (j + np.random.rand()) / ny
                ray = cam.get_ray(u, v)
                col += ray_color(ray, world, 50)
            col /= ns

            im[i, -(j + 1)] = np.round(255 * col).astype(int)

    imageio.imwrite('simple_sphere.png', im.transpose((1, 0, 2)))


if __name__ == '__main__':
    main()
