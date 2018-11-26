import unittest
from worlds.gridworld import GridWorld, Action, Item


class GridWorldTest(unittest.TestCase):

    def test_gridworld_grid_and_observation_should_conform_to_spec_at_initialization(self):
        gw: GridWorld = GridWorld(5, 8, Item.PLANK)
        obs = gw.reset()

        print(gw)

    def test_gridworld_grid_and_observation_should_update_on_step(self):
        gw: GridWorld = GridWorld(5, 8, Item.PLANK)
        obs = gw.reset()

        print(gw)
        gw.step(Action.MOVE_LEFT)
        print(gw)
        gw.step(Action.USE)
        print(gw)
        gw.step(Action.MOVE_LEFT)
        print(gw)
        obs, reward, done, info = gw.step(Action.USE)
        print(gw)

        self.assertTrue(done)

    def test_gridworld_environment_should_terminate_after_max_timesteps(self):
        gw: GridWorld = GridWorld(5, 8, Item.PLANK, max_timesteps=1)
        obs = gw.reset()

        obs, reward, done, info = gw.step(Action.MOVE_UP)

        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
