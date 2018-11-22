import random
from typing import List, Tuple, Mapping, Any
from textwrap import indent
from enum import Enum


class Action(Enum):
    STAY_PUT = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    PICK_UP = 5
    PUT_DOWN = 6


Observation = List[Tuple[bool, List[List[Tuple]]]]

# Constants.
# ----------
REWARD_GOAL = 1
MOVE_DR_DC = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}
WINDOW_RADIUS = 2


class Cell:
    """A single cell in the grid world."""

    def __init__(
            self,
            row: int,
            col: int,
            is_goal: bool = False,
            has_object: bool = False,
    ):
        self.row: int = row
        self.col: int = col
        self.is_goal: bool = is_goal
        self.has_object: bool = has_object


class Manipulator:
    """A manipulator that can pick up objects."""

    def __init__(
            self,
            row: int,
            col: int,
            has_object: bool = False,
    ):
        self.row: bool = row
        self.col: bool = col
        self.has_object: bool = has_object


class GridWorld:
    """
    Simulate a 2D grid world for pick-and-place tasks using multiple object manipulators:

        . G G . .
        . . . O .
        O . . . .
        . . M . M

    . : "EMPTY" cell.
    O : "OBJECT" that can be moved around.
    G : "GOAL" area in which agent aims to put down objects.
    M : "MANIPULATOR" that agent controls (e.g. a number of hands).

    Each MANIPULATOR has the following action space:

        MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PICK_UP, PUT_DOWN

    The world dynamics conform to the following laws:

        - A MANIPULATOR can move anywhere in the grid. Any attempt to move off the grid is a no-op.
        - A MANIPULATOR must be on top of an OBJECT to pick it up. When put down, the OBJECT is in
          the location of the MANIPULATOR.
        - A MANIPULATOR can only pick up a single OBJECT at a time. Any attempt to pick up a second
          OBJECT, or pick up / put down a non-existent OBJECT, is a no-op.
        - A cell can have at most a single OBJECT at a time, whether held by the cell or by a
          MANIPULATOR. Any attempt to move a MANIPULATOR with an OBJECT into a cell with an
          OBJECT is a no-op.
        - At every timestep, all put down actions are executed first, then pick up, then move. This
          means that it's possible for one MANIPULATOR to "hand-off" an OBJECT to another.
        - An OBJECT placed on the GOAL will vanish, and a positive unit reward will be conferred.
        - An episode is over when all OBJECTS have been consumed by GOAL areas, or at a fixed
          timestep limit.
        - At the start, OBJECTS, MANIPULATORS, and GOALS are non-overlapping.
        - A MANIPULATOR with a lower ID has precedence to act first (legally).
    """

    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            num_manipulators: int = 2,
            num_goals: int = 2,
            num_objects: int = 5,
            max_timesteps: int = 100,
            seed: int = 0,
    ):
        """Initializes the grid randomly conforming to the specifications."""
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self.num_manipulators: int = num_manipulators
        self.num_goals: int = num_goals
        self.num_objects: int = num_objects
        self.max_timesteps: int = max_timesteps

        self.seed: int = seed
        random.seed(seed)

        # Initialize empty world
        self.grid: List[List[Cell]] = []
        self.manipulators: List[Manipulator] = []

    def reset(self) -> Observation:
        """Assign manipulators, goals, and objects to initial cell positions that do not overlap."""
        self.grid = [[Cell(r, c) for c in range(self.num_cols)] for r in range(self.num_rows)]

        # Sample cell locations (r, c) without replacement
        num_points: int = self.num_manipulators + self.num_goals + self.num_objects
        points: List[Tuple[int, int]] = [
            divmod(i, self.num_cols)
            for i in random.sample(range(self.num_rows * self.num_cols), num_points)
        ]
        curr: int = 0

        # Construct manipulators
        self.manipulators = [
            Manipulator(r, c) for r, c in points[curr:curr + self.num_manipulators]
        ]
        curr += self.num_manipulators

        # Construct goal areas
        for r, c in points[curr:curr + self.num_goals]:
            self.grid[r][c].is_goal = True
        curr += self.num_goals

        # Construct objects
        for r, c in points[curr:curr + self.num_objects]:
            self.grid[r][c].has_object = True
        curr += self.num_objects

        return self._construct_observation()

    def step(self, actions: List[Action]) -> Tuple[Observation, float, bool, Mapping]:
        """Advance a single timestep given the actions of each manipulator.
        :param actions
            List of action identifiers, one for each manipulator.
        """
        observation: List = []
        reward: float = 0
        done: bool = False
        info: Mapping[str, Any] = {}

        # Perform PUT_DOWN actions
        for manipulator_id, action in enumerate(actions):
            if action == Action.PUT_DOWN:
                reward += self._action_put_down(manipulator_id)

        # Perform PICK_UP actions
        for manipulator_id, action in enumerate(actions):
            if action == Action.PICK_UP:
                self._action_pick_up(manipulator_id)

        # Perform MOVE actions
        for manipulator_id, action in enumerate(actions):
            if action in (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT):
                self._action_move(manipulator_id, action)

        # Construct observation
        observation = self._construct_observation()

        return observation, reward, done, info

    def _action_put_down(self, manipulator_id: int) -> float:
        m: Manipulator = self.manipulators[manipulator_id]
        if not m.has_object: return 0
        m.has_object = False
        if self.grid[m.row][m.col].is_goal:
            return REWARD_GOAL
        else:
            self.grid[m.row][m.col].has_object = True
            return 0

    def _action_pick_up(self, manipulator_id: int) -> None:
        m: Manipulator = self.manipulators[manipulator_id]
        if not self.grid[m.row][m.col].has_object: return
        self.grid[m.row][m.col].has_object = False
        m.has_object = True

    def _action_move(self, manipulator_id: int, action: Action) -> None:
        m: Manipulator = self.manipulators[manipulator_id]
        dr, dc = MOVE_DR_DC[action]
        r_new = m.row + dr
        c_new = m.col + dc
        if r_new < 0 or self.num_rows <= r_new: return
        if c_new < 0 or self.num_cols <= c_new: return
        if m.has_object:
            if self.grid[r_new][c_new].has_object: return
            for mid, m_other in self.manipulators:
                if mid == manipulator_id: continue
                if m_other.row == r_new and m_other.col == c_new and m_other.has_object: return
        m.row = r_new
        m.col = c_new

    def _construct_observation(self) -> Observation:
        """Returns an observation of the environment state.

        Each manipulator has a window of observation around itself (a 2 *  WINDOW_RADIUS + 1 cell
        square), so it can access whether the surrounding states are goal states or have objects, in
        addition to its own grasping state:

            [(m.has_object, [[(c.has_object, c.is_goal, c.has_) for each cell c]]) for each
            manipulator m]

        Each tuple in the observation is the state for a particular manipulator, where the
        first element is the manipulator grasping state and the second element is a window of
        surrounding cell states.
        """

        def _construct_window(m: Manipulator) -> List[List[Tuple]]:
            window = []
            for r in range(m.row - WINDOW_RADIUS, m.row + WINDOW_RADIUS + 1):
                is_row_valid = 0 <= r < self.num_rows
                row = []
                for c in range(m.col - WINDOW_RADIUS, m.col + WINDOW_RADIUS + 1):
                    is_col_valid = 0 <= c < self.num_cols
                    cell_state = (
                        self.grid[r][c].has_object,
                        self.grid[r][c].is_goal,
                    ) if (is_row_valid and is_col_valid) else (
                        False,
                        False,
                    )
                    cell_manipulators = tuple(m.row == r and m.col == c for m in self.manipulators)
                    row.append(cell_state + cell_manipulators)
                window.append(row)
            return window

        return [(m.has_object, _construct_window(m)) for m in self.manipulators]

    def __str__(self) -> str:

        def _cell_to_symbol(cell: Cell):
            if cell.is_goal: return "G"
            elif cell.has_object: return "O"
            else: return "."

        symbols = [
            [_cell_to_symbol(self.grid[r][c])
             for c in range(self.num_cols)]
            for r in range(self.num_rows)
        ]
        for m in self.manipulators:
            symbols[m.row][m.col] = "M"

        strs = []
        strs.append("\n".join(" ".join(col for col in row) for row in symbols))

        obs = self._construct_observation()
        for mid, (has_object, window) in enumerate(obs):
            strs.append("manipulator {}".format(mid))
            strs.append(indent("has_object: {}".format(has_object), " " * 4))
            strs.append(indent("window:", " " * 4))
            strs.append(
                indent(
                    "\n".join(
                        ", ".join("".join(str(int(x)) for x in c) for c in r) for r in window
                    ), " " * 8
                )
            )

        return "\n".join(strs)
