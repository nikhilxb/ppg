#from typing import List, Tuple, Mapping, Callable, Union, Any, Optional, cast
import numpy as np

class asset(object):
    """
    Constructs an asset object, which represents the manipulators controlled by the algorithm.
    One instance of this class is constructed each time a tuple is passed to the GridWorld
    constructor through the Assets list (see below).
    """
    def __init__(self, location, holding=False):
        """
        The constructor is passed a tuple indicating the asset's position in real space, and
        a flag to indicate if the asset is holding an object (defaulted to False)
        """
        self.location = location
        self.holding = holding # Boolean to indicate if asset is holding an object

    def grid_location(self, grid_size):
        """
        Compute the coordinates of the asset on the grid data structure (see below)
        """
        row = 2 # assets always given row 2
        col = self.location[0]*grid_size[1] + self.location[1]

        return (row, col)

    def print_summary(self, grid_size):
        """
        Prints a summary of the current asset (mostly for debugging)
        """
        print("Asset's real-world location " + str(self.location))
        grid_coord = self.grid_location(grid_size)
        print("Asset's grid location " + str(grid_coord))
        if self.holding:
            print("The asset is holding an object")
        else:
            print("The asset is not holding an object")

class GridWorld(object):
    """
    Basic class for an instantiation of a grid world object. Constructs a two-dimensional grid
    where each cell can take on one of four values: Empty, Asset, Object, or Goal. An "Asset" is
    an agent that the algorithm can control, with two functions: Pick Up and Drop. An "Object" is
    something that can be picked up/dropped by the Asset. A "Goal" marks a cell as part of the goal
    area.
    A simple 4x4 grid might look like:

    X G G X
    X X O X
    O X O O
    X A A X

    where "X" is an empty cell, "G" is a goal cell, "O" is an object, and "A" is an asset.

    The grid is represented in memory with a vector of cells; indexing the upper left cell
    at 0 and the bottom right cell at (n*m - 1) for an n x m grid, we have

    grid = [ cell_0,
             cell_1,
              ...,
             cell_(n*m - 1)
           ]

    Each cell_i is a one-hot vector representing what's contained in that cell:

    cell_i = [ object,
               goal,
               asset
             ]

    So a cell with an object would be cell_i = [0, 1, 0]. An empty cell would be cell_i = [0, 0, 0] (e.g. all 0's = empty).
    """

    """
    TODOS: fix command line prompts / argparser, implement a function that can map from the grid object to a pictoral
    representation of the grid (probably not high priority right now), start implementing functions that allow assets
    to move, implement basic pick up/put down functions.
    Also should start thinking about a grid world solver. Can start coding this up now since we have the basic grid world
    implementation up.
    """

    def __init__(self, Assets, Goals, Objects, Size):
        """
        Constructor for the grid world object. Arguments:
        Assets --> List of tuples giving initial locations of assets
        Objects --> List of tuples giving initial locations of objects
        Goals --> List of tuples giving initial locations of goals
        Size --> Tuple giving size of grid (row, column)
        """
        self.assets = Assets
        self.goals = Goals
        self.objs = Objects
        self.size = Size

    def construct_grid(self):
        """
        Constructs a representation of the grid discussed above.
        Returns: a 2D numpy array representing the grid world.
        Also returns a dictionary mapping assets, objects, and goals
        to their coordinates in the grid object (NOT real world).
        """
        # dictionary to map assets/objects/goals to coordinates in grid object
        locations = {}

        # compute the number of columns in the grid object
        columns = self.size[0]*self.size[1]

        # three rows: object, goal,asset
        grid = np.zeros((3, columns), dtype=int)
        all_objects = [self.objs, self.goals, self.assets]

        for index, object in enumerate(all_objects):
            for j, coord in enumerate(object):
                # this computes the object's column in the grid array
                object_col = coord[0]*self.size[1] + coord[1]
                # set the relevant index to 1
                grid[index][object_col] = 1

                # update the dictionary
                if index == 0:
                    locations["object_" + str(j)] = (index, object_col)
                elif index == 1:
                    locations["goal_" + str(j)] = (index, object_col)
                else:
                    locations["asset_" + str(j)] = (index, object_col)

        return grid, locations


"""
Accept arguments from the command line, and construct the grid object automatically / print it to the console.
TODO: Fix this (might not be really necessary rn)
"""
if __name__ == "__main__":

    from argparse import ArgumentParser
    p = ArgumentParser()

    p.add_argument("input_assets", type=list, default=[], help="Input a list of tuples indicating asset locations")
    p.add_argument("input_objs", type=list, default=[], help="Input a list of tuples indicating object locations")
    p.add_argument("input_goals", type=list, default=[], help="Input a list of tuples indicating goal locations")
    p.add_argument("input_size", type=tuple, default=[], help="Input a tuple indicating grid size")

    args = p.parse_args()

    grid_object = GridWorld(args.input_assets, args.input_goals, args.input_objs, args.input_size)
    grid, locations = GridWorld.construct_grid()

    print(grid)
    print(locations)
