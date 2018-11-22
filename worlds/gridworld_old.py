import numpy as np
from typing import List, Tuple, Mapping, Callable, Union, Any, Optional, cast


class Asset(object):
    """
    Parent class for anything on the board that can be manipulated; this includes
    the Hand and Block subclasses below. Each asset has a few common properties:
        - real-world location
        - grid column index (e.g., the column of the asset in the 2D GridWorld object stored in memory)
        - name
    In addition, the constructor is passed the grid size so the grid column can be calculated from
    the real location. Note that the grid row depends on the specific subclass, so this coordinate is
    abstracted to the subclasses.
    """

    def __init__(self, location, name, grid_size):
        self.real_location = location
        self.name = name
        self.grid_size = grid_size

        # compute the grid column; the grid row depends on the specific subclass
        self.grid_col = self.real_location[0] * grid_size[1] + self.real_location[1]


class Hand(Asset):
    """
    Constructs a Hand object, which represents the manipulators controlled by the algorithm.
    """

    def __init__(self, location, name, grid_size, holding=False):
        super(Hand, self).__init__(location, name, grid_size)
        self.holding = holding  # Boolean to indicate if hand is holding an object
        self.goal_blocks = 0  # track how many blocks this hand has moved to the goal area
        self.current_block = None  # track which block is currently held, if any

        # construct the grid location of the hand
        self.grid_row = 2  # hand always given row 2
        self.grid_location = (self.grid_row, self.grid_col)

    def print_summary(self):
        """
        Prints a summary of the current hand (mostly for debugging)
        """
        print("Hand's real-world location " + str(self.real_location))
        print("Hand's grid location " + str(self.grid_location))
        if self.holding:
            print("The hand is holding an object")
        else:
            print("The hand is not holding an object")


class Block(Asset):
    """
    Constructs a Block object, which represents an object the Hands can manipulate.
    """

    def __init__(self, location, name, grid_size, held=False):
        super(Block, self).__init__(location, name, grid_size)
        self.held = held

        # construct grid location of block
        self.grid_row = 0
        self.grid_location = (self.grid_row, self.grid_col)

    def print_summary(self):
        print("Block's real-world location " + str(self.real_location))
        print("Block's grid location " + str(self.grid_location))
        if self.held:
            print("The hand is holding an object")
        else:
            print("The hand is not holding an object")


class Goal(Asset):
    """
    Constructs a Goal object, reprenting a goal area in the grid. Right now,
    Goal objects are passive: they only exist to indicate if an object should
    be removed from the board after being dropped by a hand.
    """

    def __init__(self, location, name, grid_size):
        super(Goal, self).__init__(location, name, grid_size)
        self.grid_row = 1
        self.grid_location = (self.grid_row, self.grid_col)

    def print_summary(self):
        print("Goal's real-world location " + str(self.real_location))
        print("Goal's grid location " + str(self.grid_location))


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

    where "X" is an empty cell, "G" is a goal cell, "O" is an object, and "A" is an asset (called a Hand in the above class).

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

    def __init__(self, Hands, Goals, Blocks, Size):
        """
        Constructor for the grid world object. Arguments:
        Assets --> List of tuples giving initial locations of assets
        Objects --> List of tuples giving initial locations of objects
        Goals --> List of tuples giving initial locations of goals
        Size --> Tuple giving size of grid (row, column)
        """
        self.hands = Hands
        self.goals = Goals
        self.blocks = Blocks
        self.size = Size

        # Construct the hands, goals, bocks here, in the constructor
        # Instead of constructing a dictionary of hand/block/goal locations
        self.hand_list = []
        for i, coord_h in enumerate(self.hands):
            # construct each hand object
            hand_name = "hand_" + str(i)
            hand_to_add = Hand(coord_h, hand_name, self.size)
            self.hand_list.append(hand_to_add)

        self.block_list = []
        for j, coord_b in enumerate(self.blocks):
            # construct each block object
            block_name = "block_" + str(j)
            block_to_add = Block(coord_b, block_name, self.size)
            self.block_list.append(block_to_add)

        self.goal_list = []
        for k, coord_g in enumerate(self.goals):
            # construct each goal object
            goal_name = "goal_" + str(k)
            goal_to_add = Goal(coord_g, goal_name, self.size)
            self.goal_list.append(goal_to_add)

        # Construct the basic grid object
        columns = self.size[0] * self.size[1]
        # three rows: block, goal, hand
        self.grid = np.zeros((3, columns), dtype=int)

    def construct_grid(self):
        """
        Constructs a representation of the grid discussed above.
        Returns: a 2D numpy array representing the grid world.
        Also returns a dictionary mapping assets, objects, and goals
        to their coordinates in the grid object (NOT real world).
        """
        # dictionary to map assets/objects/goals to coordinates in grid object
        # deprecated 11/20 in favor of Hand object creation in constructor
        #locations = {}

        # add hands to the grid
        for hand in self.hand_list:
            row_h = hand.grid_row
            col_h = hand.grid_col
            self.grid[row_h][col_h] = 1

        # add blocks to the grid
        for block in self.block_list:
            row_b = block.grid_row
            col_b = block.grid_col
            self.grid[row_b][col_b] = 1

        # add goals to the grid
        for goal in self.goal_list:
            row_g = goal.grid_row
            col_g = goal.grid_col
            self.grid[row_g][col_g] = 1

    """
    Action definitions. We have six available actions for a hand:
        - move up
        - move down
        - move left
        - move right
        - pick up
        - drop
    """

    def hand_move(self, hand, direction):
        """
        Move the passed hand object one cell up / down / left / right. The flag "direction" indicates
        which way we are moving: direction = 'u' , 'd' , 'l' , or 'r'.

        The conditionals below determine (1) whether we are moving in the positive or negative direction,
        (2) where the appropriate boundary is, and (3) which index of the tuple we are changing
        """
        if direction == 'u':
            move = -1
            boundary = 0
            i = 0
        elif direction == 'd':
            move = 1
            boundary = self.size[0] - 1
            i = 0
        elif direction == 'l':
            move = -1
            boundary = 0
            i = 1
        elif direction == 'r':
            move = 1
            boundary = self.size[1] - 1
            i = 1
        else:
            # passed a non-allowed direction
            return None

        if hand.real_location[i] == boundary:
            # need a good failure condition; not sure what this will be yet
            # basically, don't move hand
            return None
        else:
            """
            We need to (1) update hand's real location, (2) update hand's grid location, and
            (3) update the grid object itself
            """
            # first, we update the old coordinates of the grid
            self.grid[hand.grid_location[0]][hand.grid_location[1]] = 0

            # update real location
            # should we use something other than a tuple, since tuples are immutable?
            new_coord = hand.real_location[i] + move
            if i == 0:
                hand.real_location = (new_coord, hand.real_location[1])
            else:
                hand.real_location = (hand.real_location[0], new_coord)

            # update grid location
            hand.grid_col = hand.real_location[0] * self.size[1] + hand.real_location[1]
            hand.grid_location = (hand.grid_row, hand.grid_col)

            # finally, update the new grid coordinates
            self.grid[hand.grid_location[0]][hand.grid_location[1]] = 1

    def find_object(self, target_coord):
        """
        Given a grid target coordinate tuple (r,c), this function finds the object in block_list (if any)
        that has the same grid coordinates. Used as a helper function for hand drop/pick below.
        """
        for block in self.block_list:
            if block.grid_location == target_coord:
                return block
        # such a block is not present
        # shouldn't be the case if initialized correctly
        return None

    def hand_pick(self, hand):
        """
        Pick up an object, if one is located at the current cell; otherwise, return None
        """
        cell_index = hand.grid_location[0]
        if self.grid[cell_index][0] == 1:
            # there is an object present
            # find the object in block_list
            cell_object = self.find_object(hand.grid_location)
            if not hand.holding and cell_object is not None:
                # change the hand's flag
                hand.holding = True
                hand.current_block = cell_object
                # reassign current element to 0
                self.grid[cell_index][0] = 0
            else:
                # currently holding an object
                return None
        else:
            # no object present at this cell
            return None

    def hand_drop(self, hand):
        """
        Drop an object, if no object located at current cell; otherwise, return None.
        Also check if object is dropped on a goal cell; if so, increment the goal_blocks
        counter and remove the object from the block_list cache.
        """
        cell_index = hand.grid_location[0]
        if self.grid[cell_index][0] == 0:
            # no object present
            if hand.holding:
                # change the hand's flag
                hand.holding = False
                if self.grid[cell_index][1] == 0:
                    # current cell is not a goal element
                    self.grid[cell_index][0] = 1
                    hand.current_block = None
                else:
                    # current cell is a goal element
                    # do not deposit block, but remove it from cache
                    hand.goal_blocks += 1
                    self.block_list.remove(hand.current_block)
                    hand.current_block = None
            else:
                # not holding an object
                return None
        else:
            # object already present at cell
            return None
