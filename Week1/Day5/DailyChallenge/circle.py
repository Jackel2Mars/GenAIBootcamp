import math

class Circle:
    def __init__(self, *, radius=None, diameter=None):
        if (radius is not None) and (diameter is not None):
            raise ValueError("Specify either radius or diameter, not both.") #I decided to refuse when sent both radius and diameter
        elif radius is not None:
            self.radius = radius
        elif diameter is not None:
            self.radius = diameter / 2
        else:
            raise ValueError("You must specify either radius or diameter.") #raise an error if no attribute is sent on instanciation

    @property
    def diameter(self):
        return self.radius * 2 #method is declared with decorator so that diameter can be called as attribute

    #Compute the circle’s area
    def area(self):
        return math.pi * self.radius ** 2
    
    #Print the attributes of the circle - use a dunder method
    def __str__(self):
        return f"Circle(radius={self.radius:.2f}, diameter={self.diameter:.2f})"

    #Be able to add two circles together, and return a new circle with the new radius - use a dunder method
    def __add__(self, other):
        return Circle(radius=self.radius + other.radius)

    #Be able to compare two circles to see which is bigger, and return a Boolean - use a dunder method
    def __gt__(self, other):
        return self.radius > other.radius

    #Be able to compare two circles and see if there are equal, and return a Boolean- use a dunder method
    def __eq__(self, other):
        return self.radius == other.radius
    
    #Be able to put them in a list and sort them
    def sorted_circles(self, other):
        return sorted([self, other], key=lambda c: c.radius)
    
