import math

class Circle:
    def __init__(self, *, radius=None, diameter=None):
        if (radius is not None) and (diameter is not None):
            raise ValueError("Specify either radius or diameter, not both.")
        elif radius is not None:
            self._radius = radius
        elif diameter is not None:
            self._radius = diameter / 2
        else:
            raise ValueError("You must specify either radius or diameter.")

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2

    @property
    def area(self):
        return math.pi * self._radius ** 2

    def __str__(self):
        return f"Circle(radius={self.radius}, diameter={self.diameter}, area={self.area:.2f})"

    def __add__(self, other):
        return Circle(radius=self.radius + other.radius)

    def __gt__(self, other):
        return self.radius > other.radius

    def __eq__(self, other):
        return self.radius == other.radius
    
    def list_sorted(self, other):
        return sorted([self, other], key=lambda c: c.radius)
