import numpy as np
from ray import Ray
from utils import unit_vector, random_in_unit_disk


class Camera(object):

    def __init__(self, lookfrom, lookat, vup, vfov, aspect, aperture, focus_dist):
        """
        Initialize camera position, orientation, field of view and aperture.

        Args:
            lookfrom: camera location within scene
            lookat: coordinates towards which camera is oriented
            vup: "up" direction
            vfov: "field of view" angle, in rad
            aspect: aspect ratio (width / height) of focus window
            aperture: aperture (diameter) of camera lense
            focus_dist: distance of focus plane from camera
        """
        self._lens_radius = aperture / 2
        # vfov is top to bottom in rad
        half_height = np.tan(vfov/2)
        half_width = aspect * half_height
        self._origin = lookfrom
        # orthonormal basis
        self._w = unit_vector(lookfrom - lookat)
        self._u = unit_vector(np.cross(vup, self._w))
        self._v = np.cross(self._w, self._u)
        # define the focus plane window
        self._lower_left_corner = self._origin \
            - half_width *focus_dist*self._u   \
            - half_height*focus_dist*self._v   \
            -             focus_dist*self._w
        self._horizontal = 2*half_width *focus_dist*self._u
        self._vertical   = 2*half_height*focus_dist*self._v

    def get_ray(self, s, t):
        """
        Get a ray originating from a random position on the lense (to imitate
        depth of field), targeting the focus window at relative coordinates.

        Args:
            s: relative x-coordinate within focus window (real number between 0 and 1)
            t: relative y-coordinate within focus window (real number between 0 and 1)

        Returns:
            Ray: generated ray
        """
        rd = self._lens_radius * random_in_unit_disk()
        offset = rd[0]*self._u + rd[1]*self._v
        ray_origin = self._origin + offset
        direction = self._lower_left_corner + s*self._horizontal + t*self._vertical - ray_origin
        return Ray(ray_origin, direction)
