"""Exercises XP
Exercise 1: Pets

Key Python Topics:

Inheritance
Class instantiation
Lists
Polymorphism


Instructions:

Use the provided Pets and Cat classes to create a Siamese breed, instantiate cat objects, and use the Pets class to manage them.
See the example below, before diving in.



Step 1: Create the Siamese Class

Create a class called Siamese that inherits from the Cat class.
You can add any specific attributes or methods for the Siamese breed, or leave it as is if there are no unique behaviors. we should tell them what to add, maybe a new attribute or methods


Step 2: Create a List of Cat Instances

Create a list called all_cats that contains instances of Bengal, Chartreux, and Siamese cats.
Example: all_cats = [bengal_obj, chartreux_obj, siamese_obj]
Give each cat a name and age.


Step 3: Create a Pets Instance

Create an instance of the Pets class called sara_pets, passing the all_cats list as an argument.


Step 4: Take Cats for a Walk

Call the walk() method on the sara_pets instance.
This should print the result of calling the walk() method on each cat in the list.


Example:

class Pets():
    def __init__(self, animals):
        self.animals = animals

    def walk(self):
        for animal in self.animals:
            print(animal.walk())

class Cat():
    is_lazy = True

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def walk(self):
        return f'{self.name} is just walking around'

class Bengal(Cat):
    def sing(self, sounds):
        return f'{sounds}'

class Chartreux(Cat):
    def sing(self, sounds):
        return f'{sounds}'


Step 1: Create the Siamese class

Step 2: Create a list of cat instances

Step 3: Create a Pets instance of the list of cat instances

sara_pets = Pets(all_cats)

Step 4: Take cats for a walk
sara_pets.walk()"""
class Pets():
    def __init__(self, animals):
        self.animals = animals

    def walk(self):
        for animal in self.animals:
            print(animal.walk())

class Cat():
    is_lazy = True

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def walk(self):
        return f'{self.name} is just walking around'

class Bengal(Cat):
    def sing(self, sounds):
        return f'{sounds}'

class Chartreux(Cat):
    def sing(self, sounds):
        return f'{sounds}'

#Step 1: Create the Siamese class
class Siamese(Cat):
    def sing(self, sounds):
        return f'{sounds}'

#Step 2: Create a list of cat instances
all_cats = [Bengal("Kitty", 3), Chartreux("Selena", 4), Siamese("Felicia", 5)]

#Step 3: Create a Pets instance of the list of cat instances
sara_pets = Pets(all_cats)

#Step 4: Take cats for a walk   
sara_pets.walk()

"""Exercise 2: Dogs

Goal: Create a Dog class with methods for barking, running speed, and fighting.



Key Python Topics:

Classes and objects
Methods
Attributes


Instructions:

Step 1: Create the Dog Class

Create a class called Dog with name, age, and weight attributes.
Implement a bark() method that returns “ is barking”.
Implement a run_speed() method that returns weight / age * 10.
Implement a fight(other_dog) method that returns a string indicating which dog won the fight, based on run_speed * weight.


Step 2: Create Dog Instances

Create three instances of the Dog class with different names, ages, and weights.


Step 3: Test Dog Methods

Call the bark(), run_speed(), and fight() methods on the dog instances to test their functionality.


Example (Conceptual, No Direct Solution):

class Dog:
    def __init__(self, name, age, weight):
        # ... code to initialize attributes ...

    def bark(self):
        # ... code to return bark message ...

    def run_speed(self):
        # ... code to return run speed ...

    def fight(self, other_dog):
        # ... code to determine and return winner ...

# Step 2: Create dog instances
#... your code here

# Step 3: Test dog methods
print(dog1.bark())
print(dog2.run_speed())
print(dog1.fight(dog2))"""

#Step 1: Create the Dog Class
class Dog:
    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def bark(self):
        return f"{self.name} is barking"

    def run_speed(self):
        return self.weight/self.age * 10

    def fight(self, other_dog):
        if self.run_speed() * self.weight > other_dog.run_speed() * other_dog.weight:
            return f"{self.name} won its fight against {other_dog.name}"
        elif self.run_speed() * self.weight < other_dog.run_speed() * other_dog.weight:
            return f"{self.name} lost its fight against {other_dog.name}"
        else:
            return f"The fight between {self.name} and {other_dog.name} was a tie"

# Step 2: Create dog instances
dog1 = Dog("Knuckles", 3, 28)
dog2 = Dog("Clyde", 4, 19)
dog3 = Dog("Krypto", 5, 31)

# Step 3: Test dog methods
print(dog1.bark())
print(dog2.run_speed())
print(dog1.fight(dog2))
