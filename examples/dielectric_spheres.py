from __future__ import division
import numpy as np
import imageio
import sys
sys.path.append('../engine/')
from surface import SurfaceAssembly, Sphere
from material import Lambertian, Metal, Dielectric
from camera import Camera
from utils import unit_vector


def color(ray, world, depth):
    """Perform ray tracing and return color of ray."""
    rec, _ = world.hit(ray, 0.001, 1e6)
    if rec is not None:
        scattered, attenuation = rec.material.scatter(ray, rec)
        if depth > 0 and scattered is not None:
             return attenuation * color(scattered, world, depth - 1)
        else:
            return np.zeros(3)
    else:
        # blue background sky
        unitdir = unit_vector(ray.direction)
        t = 0.5*(unitdir[1] + 1)
        return (1 - t)*np.array([1.0, 1.0, 1.0]) + t*np.array([0.5, 0.7, 1.0])


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

    # define world geometry
    world = SurfaceAssembly()
    world.add_object(Sphere(np.array([ 0., 0., -1.]), 0.5, Lambertian(np.array([0.1, 0.2, 0.5]))))
    world.add_object(Sphere(np.array([ 1., 0., -1.]), 0.5, Metal(np.array([0.8, 0.6, 0.2]), 1.0)))
    # imitate hollow glass sphere
    world.add_object(Sphere(np.array([-1., 0., -1.]),  0.5,  Dielectric(1.5)))
    world.add_object(Sphere(np.array([-1., 0., -1.]), -0.45, Dielectric(1.5)))
    # large sphere imitating ground floor
    world.add_object(Sphere(np.array([0., -100.5, -1.]), 100., Lambertian(np.array([0.8, 0.8, 0.0]))))

    # fill image pixels
    im = np.zeros((nx, ny, 3), dtype=np.uint8)
    for i in range(nx):
        for j in range(ny):
            col = np.zeros(3)
            for s in range(ns):
                u = (i + np.random.rand()) / nx
                v = (j + np.random.rand()) / ny
                ray = cam.get_ray(u, v)
                col += color(ray, world, 50)
            col /= ns

            # take sqrt for gamma correction
            im[i, -(j + 1)] = np.round(255 * np.sqrt(col)).astype(int)

    imageio.imwrite('dielectric_spheres.png', im.transpose((1, 0, 2)))


if __name__ == '__main__':
    main()
