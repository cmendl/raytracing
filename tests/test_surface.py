import unittest
import numpy as np
import sys
sys.path.append('../engine/')
from surface import Sphere
from ray import Ray


class TestSurface(unittest.TestCase):

    def test_sphere_hit(self):

        sphere = Sphere(np.array([0.2, -0.5, -1.4]), 0.8, None)

        # ray specified by origin and direction
        ray = Ray(np.array([0.3, 0.1, 0.4]), np.array([-0.1, -0.3, -1.]))

        # intersect ray with sphere
        rec, t = sphere.hit(ray, 0.1, 10.)

        self.assertAlmostEqual(np.linalg.norm(ray.point_at_parameter(t) - rec.point), 0, delta=1e-14,
            msg='hit record must be consistent with ray parameter')

        self.assertAlmostEqual(abs(np.linalg.norm(rec.point - sphere.center) - sphere.radius), 0, delta=1e-14,
            msg='distance between hit point and sphere center must be equal to sphere radius')


if __name__ == '__main__':
    unittest.main()
