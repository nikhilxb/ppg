import unittest
from worlds.gridworld import GridWorld


class GridWorldTest(unittest.TestCase):

    def test_gridworld_initializes_correctly(self):
        gw: GridWorld = GridWorld(5, 8, num_manipulators=2, num_goals=3, num_objects=5)
        obs = gw.reset()

        print(gw)


if __name__ == "__main__":
    unittest.main()
