
class Ray(object):
    """
    Light ray class, storing origin and direction of the ray.
    The direction needs not be normalized.
    """

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        """Compute point on ray at parameter `t`."""
        return self.origin + t*self.direction
