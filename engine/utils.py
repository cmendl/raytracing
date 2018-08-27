import numpy as np


def unit_vector(v):
    """"Normalize vector `v`."""
    n = np.linalg.norm(v)
    if n > 0:
        return v / n
    else:
        # zero vector
        return v


def random_in_unit_disk():
    """Generate a uniformly random point within the unit disk."""
    while True:
        p = 2 * np.random.rand(2) - 1
        if np.dot(p, p) < 1:
            return p


def random_in_unit_sphere():
    """Generate a uniformly random point within the unit sphere."""
    while True:
        p = 2 * np.random.rand(3) - 1
        if np.dot(p, p) < 1:
            return p
