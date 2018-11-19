"""
Playground for testing the functionality of the GridWorld class
"""

goals = [(0,1)]
assets = [(1,1)]
objects = []
size = (2,2)

from grid_world import GridWorld
test = GridWorld(assets, goals, objects, size)

grid_1, locations_1 = test.construct_grid()

print(grid_1)
print(locations_1)
