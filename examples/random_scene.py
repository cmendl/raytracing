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
    nx = 1200
    ny =  800

    # number of samples (rays) per pixel
    ns = 10

    # define camera
    lookfrom = np.array([13., 2., 3.])
    lookat = np.zeros(3)
    vfov = np.pi/9
    aperture = 0.1
    focus_dist = 10.0
    cam = Camera(lookfrom, lookat, np.array([0., 1., 0.]), vfov, nx / ny, aperture, focus_dist)

    # define scene geometry
    scene = SurfaceAssembly()
    # three large spheres
    scene.add_object(Sphere(np.array([ 4., 1., 0.]), 1.0, Metal(np.array([0.7, 0.6, 0.5]), 0.)))
    scene.add_object(Sphere(np.array([ 0., 1., 0.]), 1.0, Dielectric(1.5)))
    scene.add_object(Sphere(np.array([-4., 1., 0.]), 1.0, Lambertian(np.array([0.4, 0.2, 0.1]))))
    # smaller spheres with random parameters
    for a in range(-11, 12):
        for b in range(-11, 12):
            center = np.array([a + 0.9*np.random.rand(), 0.2 + 0.1*np.random.rand(), b + 0.9*np.random.rand()])
            # random choice between diffusive, metal or dielectric (glass)
            choose_mat = np.random.choice(3, p=[0.8, 0.15, 0.05])
            if choose_mat == 0:
                # diffusive
                scene.add_object(Sphere(center, 0.2, Lambertian(np.random.triangular(0., 0.2, 1., size=3))))
            elif choose_mat == 1:
                # metal
                scene.add_object(Sphere(center, 0.2, Metal(0.5*(1 + np.random.rand(3)), 0.5*np.random.rand())))
            else:
                # dielectric (glass)
                scene.add_object(Sphere(center, 0.2, Dielectric(1.5)))
    # huge sphere imitating ground floor
    scene.add_object(Sphere(np.array([0., -1000., 0.]), 1000., Lambertian(np.array([0.5, 0.5, 0.5]))))

    # render image
    im = render_image(nx, ny, ns, scene, cam)

    imageio.imwrite('random_scene.png', im.transpose((1, 0, 2)))


if __name__ == '__main__':
    main()
