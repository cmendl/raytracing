import numpy as np


class HitRecord(object):
    """
    Store a "hit record" for a ray intersecting a geometric object.
    """

    def __init__(self, point, normal, material):
        # intersection point
        self.point = point
        # surface normal at intersection point
        assert abs(np.linalg.norm(normal) - 1) < 1e-13, 'hit record normal must be normalized'
        self.normal = normal
        # reference to material
        self.material = material
