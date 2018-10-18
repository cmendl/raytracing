import numpy as np
from abc import ABCMeta, abstractmethod
from ray import Ray
from utils import unit_vector, random_in_unit_sphere


class Material(object):
    __metaclass__  = ABCMeta

    @abstractmethod
    def scatter(self, ray, rec):
        """"
        Compute scattered ray and color attenuation factors for a ray hitting a surface material.

        Args:
            ray: incoming ray
            rec: corresponding hit record

        Returns:
            tuple: tuple containing
              - scattered: scattered ray
              - albedo:    reflectance per color channel
        """


class Lambertian(Material):
    """"
    Lambertian surface (ideal diffusive reflection),
    specified by albedo (reflectance) per color channel.
    """

    def __init__(self, a):
        self._albedo = a

    def scatter(self, _, rec):
        scattered = Ray(rec.point.copy(), rec.normal + random_in_unit_sphere())
        return (scattered, self._albedo)


class Metal(Material):
    """"
    Metal surface, specified by albedo (reflectance) per color channel
    and fuzziness factor (scales random additive permutation of reflected ray).
    """

    def __init__(self, a, f):
        self._albedo = a
        self._fuzz = min(f, 1)

    def scatter(self, ray, rec):
        nraydir = unit_vector(ray.direction)
        reflected = reflect(nraydir, rec.normal)
        scattered = Ray(rec.point.copy(), reflected + self._fuzz*random_in_unit_sphere())
        if np.dot(scattered.direction, rec.normal) > 0:
            return (scattered, self._albedo)
        else:
            return (None, self._albedo)


class Dielectric(Material):
    """
    Dielectric surface, specified by ratio of the indices of refraction.
    """

    def __init__(self, ri):
        self._ref_idx = ri

    def scatter(self, ray, rec):

        # normalized ray direction
        nraydir = unit_vector(ray.direction)

        reflected = reflect(nraydir, rec.normal)

        cosine = np.dot(nraydir, rec.normal)
        if cosine > 0:
            refracted = refract(nraydir, -rec.normal, self._ref_idx)
        else:
            refracted = refract(nraydir, rec.normal, 1.0 / self._ref_idx)
            cosine = -cosine

        if refracted is not None:
            reflect_prob = schlick(cosine, self._ref_idx)
        else:
            reflect_prob = 1.0

        # randomly choose between reflection or refraction
        if np.random.rand() < reflect_prob:
            return (Ray(rec.point.copy(), reflected), np.ones(3))
        else:
            return (Ray(rec.point.copy(), refracted), np.ones(3))


def reflect(v, n):
    """Reflect direction `v` at plane with normal `n`."""
    assert abs(np.linalg.norm(n) - 1) < 1e-11, 'surface normal must be normalized'
    return v - 2*np.dot(v, n)*n


def refract(v, n, ni_over_nt):
    """
    Compute direction of refracted ray according to Snell's law,
    or return None if no solution exists.
    """
    assert abs(np.linalg.norm(v) - 1) < 1e-11, 'input ray direction must be normalized'
    assert abs(np.linalg.norm(n) - 1) < 1e-11, 'surface normal must be normalized'
    dt = np.dot(v, n)
    discriminant = 1 - ni_over_nt**2 * (1 - dt**2)
    if discriminant > 0:
        return ni_over_nt*(v - n*dt) - np.sqrt(discriminant)*n
    else:
        return None


def schlick(cosine, ref_idx):
    """Schlick's approximation of specular reflection coefficient."""
    r0 = ((1 - ref_idx) / (1 + ref_idx))**2
    return r0 + (1 - r0) * (1 - cosine)**5
