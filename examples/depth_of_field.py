from __future__ import division
import numpy as np
import imageio
import sys
sys.path.append('../engine/')
from surface import SurfaceAssembly, Sphere
from material import Lambertian, Metal, Dielectric
from camera import Camera
from rendering import render_image


def main():

    # image dimensions (pixel)
    nx = 200
    ny = 100

    # number of samples (rays) per pixel
    ns = 100

    # define camera
    lookfrom = np.array([-3., 3., 3.])
    lookat = np.array([0., 0., 0.])
    vfov = np.pi/9
    aperture = 1.
    focus_dist = np.linalg.norm(lookat - lookfrom)
    cam = Camera(lookfrom, lookat, np.array([0., 1., 0.]), vfov, nx / ny, aperture, focus_dist)

    # define scene geometry
    scene = SurfaceAssembly()
    scene.add_object(Sphere(np.array([ 0., 0., 0.]), 0.5, Lambertian(np.array([0.1, 0.2, 0.5]))))
    scene.add_object(Sphere(np.array([ 1., 0., 0.]), 0.5, Metal(np.array([0.8, 0.6, 0.2]), 1.0)))
    # imitate hollow glass sphere
    scene.add_object(Sphere(np.array([-1., 0., 0.]),  0.5,  Dielectric(1.5)))
    scene.add_object(Sphere(np.array([-1., 0., 0.]), -0.45, Dielectric(1.5)))
    # large sphere imitating ground floor
    scene.add_object(Sphere(np.array([0., -100.5, 0.]), 100., Lambertian(np.array([0.8, 0.8, 0.0]))))

    # render image
    im = render_image(nx, ny, ns, scene, cam)

    imageio.imwrite('depth_of_field.png', im.transpose((1, 0, 2)))


if __name__ == '__main__':
    main()
