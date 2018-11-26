"""
This gridworld is based loosely on ideas and code used in the paper "Modular Multitask
Reinforcement Learning with Policy Sketches", by Andreas et al.
"""
import random
import crayons
from typing import List, Tuple, Mapping, Any, Iterable, Optional
from textwrap import indent
from collections import defaultdict
from enum import Enum


class Action(Enum):
    """
    Basic actions a manipulator can perform in the grid world.
    """
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    USE = 4


class Cell(Enum):
    """
    Elements of the grid world environment. Consists of immovable objects (boundaries,
    work stations) and movable "primivite" items (iron, grass, wood).
    Integers correspond to the object's index in the tensor representation of the grid world.
    """
    # Boundary cell.
    BOUNDARY = 0

    # Workstation cells.
    TOOLSHED = 1
    WORKBENCH = 2
    FACTORY = 3

    # Primitive item cells.
    IRON = 4
    GRASS = 5
    WOOD = 6


class Item(Enum):
    """
    All possible objects that can be carried by a manipulator: primitive items from the above class
    (iron, grass, wood) and composite items constructed by USING work stations and primitive items.
    """
    # Primitive items.
    IRON = 0
    GRASS = 1
    WOOD = 2

    # Composite items.
    PLANK = 3
    STICK = 4
    CLOTH = 5
    ROPE = 6
    BRIDGE = 7
    SHEARS = 8
    AXE = 9
    BED = 10
    LADDER = 11


# Constants.
# =============================================================================================

# Type definition for agent observation at every timestep.
Observation = Tuple[Mapping[Item, int], List[List[Optional[Cell]]]]

# Defines basic objects and work stations necessary to construct composite objects.
RECIPES = {
    Item.PLANK: (Cell.TOOLSHED, [Item.WOOD]),
    Item.STICK: (Cell.WORKBENCH, [Item.WOOD]),
    Item.CLOTH: (Cell.FACTORY, [Item.GRASS]),
    Item.ROPE: (Cell.TOOLSHED, [Item.GRASS]),
    Item.BRIDGE: (Cell.FACTORY, [Item.IRON, Item.WOOD]),
    Item.SHEARS: (Cell.WORKBENCH, [Item.STICK, Item.IRON]),
    Item.AXE: (Cell.TOOLSHED, [Item.STICK, Item.IRON]),
    Item.BED: (Cell.WORKBENCH, [Item.PLANK, Item.GRASS]),
    Item.LADDER: (Cell.FACTORY, [Item.PLANK, Item.STICK]),
}

# Reward for producing goal item.
REWARD_GOAL = 1

# How many cells, in each direction around agent, to include in agent observation.
WINDOW_RADIUS = 2

# Change in row, col for each move action.
MOVE_DR_DC = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}

# Map from primitive item in inventory to primitive item cell in environment.
PRIMITIVE_ITEM_TO_CELL = {
    Item.IRON: Cell.IRON,
    Item.GRASS: Cell.GRASS,
    Item.WOOD: Cell.WOOD,
}
PRIMITIVE_CELL_TO_ITEM = {v: k for k, v in PRIMITIVE_ITEM_TO_CELL.items()}

# Map from cells in environment to symbols in pretty-print representation.
CELL_TO_SYMBOL = {
    None: ".",
    Cell.BOUNDARY: "*",
    Cell.TOOLSHED: str(crayons.red("T")),
    Cell.WORKBENCH: str(crayons.red("W")),
    Cell.FACTORY: str(crayons.red("F")),
    Cell.IRON: str(crayons.green("I")),
    Cell.GRASS: str(crayons.green("G")),
    Cell.WOOD: str(crayons.green("W")),
}


class Agent:
    """An agent that can move around in the environment, add items to its inventory, and use
    produce new items at workstations using existing items."""

    def __init__(
            self,
            row: int,
            col: int,
    ):
        self.row: bool = row
        self.col: bool = col
        self.inventory: Mapping[Item, int] = defaultdict(int)


class GridWorld:
    """
    Simulate a 2D grid world for a crafting environment:

        * * * * * * * *
        * . . . . . . *
        * . . . . . . *
        * . . . T W A *
        * * * * * * * *

    . : EMPTY cell. Traversable.
    * : BOUNDARY cell. Untraversable.
    A : AGENT. Moves around in environment, consuming items and producing new items. (Blue in console)
    T, W, F : Toolshed, workbench, factory cells. WORKSTATION cells are unconsumable and are used by
        the agent to produce new composite items from existing items in its inventory. (Red in console)
    I, G, W : Iron, grass, wood cells. PRIMITIVE item cells are consumable and add the primitive
        item to the agent's inventory. (Green in console; note that workstation "W" and primitive "W" are colored differently)

    The agent has the following action space:

        MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, USE

    The world dynamics conform to the following laws:

        - An AGENT can move anywhere in the grid, except a BOUNDARY cell. Any attempt to move onto
          a BOUNDARY cell the grid is a no-op.
        - An AGENT must be on top of an PRIMITIVE cell to consume the item.
        - An AGENT can add an arbitrary number of primitive and composite objects to its inventory.
        - A cell can have at most one type (EMPTY, BOUNDARY, WORKSTATION, PRIMITIVE).
        - At the start, the AGENT, PRIMITIVE cells, and WORKSTATION cells are non-overlapping.
        - An episode is over when the GOAL item is in the AGENT's inventory, or at a fixed
          timestep limit.
        - A positive reward is conferred when the GOAL item is produced.
    """

    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            goal: Item,
            max_timesteps: int = 100,
            seed: int = 0,
    ):
        """Initializes the grid randomly conforming to the specifications."""
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self.max_timesteps: int = max_timesteps
        self.goal: Item = goal

        self.seed: int = seed
        random.seed(seed)

        # Initialize empty world
        self.grid: List[List[Cell]] = []
        self.agent: Agent = None
        self.timestep: int = 0

    def reset(self) -> Observation:
        """Assign agent and cells to initial cell positions that do not overlap."""
        self.grid = [[None for c in range(self.num_cols)] for r in range(self.num_rows)]
        self.timestep = 0

        # Sample cell locations (r, c) without replacement.
        def sample_empty_cell():
            empty_cell = None
            while empty_cell is None:
                r, c = random.randrange(self.num_rows), random.randrange(self.num_cols)
                if self.grid[r][c] is None: empty_cell = (r, c)
            return empty_cell

        # Construct boundary
        for r in range(self.num_rows):
            self.grid[r][0] = Cell.BOUNDARY
            self.grid[r][self.num_cols - 1] = Cell.BOUNDARY
        for c in range(self.num_cols):
            self.grid[0][c] = Cell.BOUNDARY
            self.grid[self.num_rows - 1][c] = Cell.BOUNDARY

        # Construct agent
        r, c = sample_empty_cell()
        self.agent = Agent(r, c)

        # Construct cells and items
        all_workstations: Set[Cell] = set()
        all_primitives: List[Cell] = []

        def get_all_primitives(item: Item) -> None:
            # Base case: Item is primitive.
            if item not in RECIPES:
                all_primitives.append(PRIMITIVE_ITEM_TO_CELL[item])
                return

            # Recursive case: Item is composite.
            workstation, subitems = RECIPES[item]
            all_workstations.add(workstation)
            for subitem in subitems:
                get_all_primitives(subitem)

        # Add all primitive items/workstations to the grid
        get_all_primitives(self.goal)
        for w in all_workstations:
            r, c = sample_empty_cell()
            self.grid[r][c] = w
        for i in all_primitives:
            r, c = sample_empty_cell()
            self.grid[r][c] = i

        return self._construct_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Mapping]:
        """
        Advance a single timestep given the actions of the agent.
        """
        self.timestep += 1

        # Dispatch on action type
        if action is Action.USE:
            self._action_use()
        elif action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
            self._action_move(action)

        # Construct returns
        observation: List = self._construct_observation()
        reward: float = REWARD_GOAL if (self.goal in self.agent.inventory) else 0
        done: bool = (self.goal in self.agent.inventory) or (self.timestep >= self.max_timesteps)
        info: Mapping[str, Any] = {}
        info["timestep"] = self.timestep

        return observation, reward, done, info

    def _action_use(self) -> None:
        """
        Pick up or use primitive items and workstations found in gridworld.
        """
        r, c = self.agent.row, self.agent.col
        cell = self.grid[r][c]
        if cell is None: return
        if cell in (Cell.IRON, Cell.GRASS, Cell.WOOD):
            # Current cell contains a primitive item. Consume it and add to inventory.
            self.agent.inventory[PRIMITIVE_CELL_TO_ITEM[cell]] += 1
            self.grid[r][c] = None
        elif cell in (Cell.TOOLSHED, Cell.WORKBENCH, Cell.FACTORY):
            # Current cell is a workstation. Consider only products producable at this specific
            # workstation, and produce the product if all ingredients are available.
            for product, (workcell, ingredients) in RECIPES.items():
                if cell is not workcell: continue
                if not all(self.agent.inventory[i] > 0 for i in ingredients): continue
                for i in ingredients:
                    self.agent.inventory[i] -= 1
                self.agent.inventory[product] += 1

    def _action_move(self, action: Action) -> None:
        """
        Move the agent up/down/left/right.
        """
        dr, dc = MOVE_DR_DC[action]
        r_new = self.agent.row + dr
        c_new = self.agent.col + dc
        if self.grid[r_new][c_new] is Cell.BOUNDARY: return
        self.agent.row = r_new
        self.agent.col = c_new

    def _construct_observation(self) -> Observation:
        """
        Returns an observation of the environment state.

        The agent has a window of observation around itself (a 2 *  WINDOW_RADIUS + 1 cell
        square), so it can access whether the surrounding cells contain primitive items or workstations.

        Returns the agent's inventory of primitive or composite items, and a matrix representation of the
        agent's observation window that can be printed to the console/used to update the agent's next actions.
        """

        window = []
        for r in range(self.agent.row - WINDOW_RADIUS, self.agent.row + WINDOW_RADIUS + 1):
            is_row_valid = 0 <= r < self.num_rows
            row = []
            for c in range(self.agent.col - WINDOW_RADIUS, self.agent.col + WINDOW_RADIUS + 1):
                is_col_valid = 0 <= c < self.num_cols
                cell_state: Optional[Cell] = self.grid[r][c] if (
                    is_row_valid and is_col_valid
                ) else None
                row.append(cell_state)
            window.append(row)

        return self.agent.inventory, window

    def __str__(self) -> str:

        symbols = [
            [CELL_TO_SYMBOL[self.grid[r][c]]
             for c in range(self.num_cols)]
            for r in range(self.num_rows)
        ]

        symbols[self.agent.row][self.agent.col] = str(crayons.blue("A"))

        strs = [
            "timestep: {}".format(self.timestep),
            "grid:",
            indent("\n".join(" ".join(col for col in row) for row in symbols), " " * 4),
        ]

        inventory, window = self._construct_observation()
        strs.append("agent:")
        strs.append(indent("inventory:", " " * 4))
        strs.append(
            indent(
                "\n".join("{}: {}".format(k, v) for k, v in inventory.items() if v > 0)
                if len(inventory) > 0 else "None", " " * 8
            )
        )
        strs.append(indent("window:", " " * 4))
        strs.append(
            indent("\n".join(" ".join(CELL_TO_SYMBOL[c] for c in r) for r in window), " " * 8)
        )

        return "\n".join(strs)
