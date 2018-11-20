
# Playground for testing the functionality of the GridWorld class

goals = [(0,1)]
hands = [(0,0)]
blocks = []
size = (2,2)

from grid_world import GridWorld
test = GridWorld(hands, goals, blocks, size)

test.construct_grid()
grid_1 = test.grid

print("/////////////////////")

print(grid_1)

hand = test.hand_list[0]
print("Hand's real coordinates before move:")
print(hand.real_location)
print("Hand's grid coordinates before move:")
print(hand.grid_location)

print("----------------------")

test.hand_move(hand, 'l')
print("Hand's real coordinates after move:")
print(hand.real_location)
print("Hand's grid coordinates after move:")
print(hand.grid_location)

print(grid_1)

# Playground for testing how classes interact

"""
from class_test import foo_1, foo_2

count = 3

test_1 = foo_2(count)

print(test_1.test_list)

"""
