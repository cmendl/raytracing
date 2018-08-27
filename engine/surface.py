import numpy as np
from abc import ABCMeta, abstractmethod
from hit_record import HitRecord


class Surface(object):
    __metaclass__  = ABCMeta

    @abstractmethod
    def hit(self, ray, t_min, t_max):
        """
        Return a hit record of a ray intersecting a surface,
        or None if there is no intersection.
        The ray parameter must be between `t_min` and `t_max`.
        """


class SurfaceAssembly(Surface):
    """
    Specify a surface as list of geometric objects.
    """

    def __init__(self):
        self._objects = []

    def add_object(self, obj):
        self._objects.append(obj)

    def hit(self, ray, t_min, t_max):
        """
        Obtain the closest hit record for a ray intersecting the stored objects.
        """
        # hit record
        rec = None
        closest_so_far = t_max
        for obj in self._objects:
            currec, t = obj.hit(ray, t_min, closest_so_far)
            if currec is not None:
                closest_so_far = t;
                rec = currec
        return (rec, closest_so_far)


class Sphere(Surface):
    """
    Geometric sphere surface.
    """

    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray, t_min, t_max):
        """
        Obtain the hit record for a ray intersecting the sphere.
        """
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2;
        discriminant = b**2 - a*c
        if discriminant > 0:
            # solutions of the quadratic equation
            t1 = -(b + np.sign(b)*np.sqrt(discriminant)) / a
            t2 = c / (a * t1)
            # smaller solution first
            for t in sorted([t1, t2]):
                if t_min <= t and t < t_max:
                    point = ray.point_at_parameter(t)
                    normal = (point - self.center) / self.radius
                    return (HitRecord(point, normal, self.material), t)
        return (None, t_max)
