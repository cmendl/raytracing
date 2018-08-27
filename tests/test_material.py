import unittest
import numpy as np
import sys
sys.path.append('../engine/')
from ray import Ray
from hit_record import HitRecord
from material import Dielectric
from utils import unit_vector


class TestMaterial(unittest.TestCase):

    def test_dielectric_scatter(self):

        dielmat = Dielectric(0.2)

        # ray specified by origin and direction
        ray = Ray(np.array([0.2, 0., 0.3]), np.array([0.1, -0.05, -1.1]))

        # hit record
        point = np.array([0.3, 0.4, -0.5])
        normal = unit_vector(np.array([0.2, 0.9, -0.1]))
        rec = HitRecord(point, normal, dielmat)

        scattered, att = dielmat.scatter(ray, rec)

        self.assertEqual(np.linalg.norm(scattered.origin - point), 0,
            msg='origin of scattered ray must agree with hit record point')

        # reference directions of scattered ray (can be either reflected or refracted)
        reflect_ref = np.array([0.054686541501393009, -0.20612619488986597, -0.97699609720757918])
        refract_ref = np.array([0.22585143494322246,   0.92588833091527412, -0.30285628276298776])
        err_reflect = np.linalg.norm(scattered.direction - reflect_ref)
        err_refract = np.linalg.norm(scattered.direction - refract_ref)
        self.assertAlmostEqual(min(err_reflect, err_refract), 0, delta=1e-14,
            msg='direction of scattered ray must agree with reference')


if __name__ == '__main__':
    unittest.main()
