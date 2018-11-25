import unittest
from worlds.gridworld import GridWorld, Action


class GridWorldTest(unittest.TestCase):

    def test_gridworld_grid_and_observation_should_conform_to_spec_at_initialization(self):
        gw: GridWorld = GridWorld(5, 8, num_manipulators=2, num_goals=3, num_objects=5)
        obs = gw.reset()

        print(gw)

    def test_gridworld_grid_and_observation_should_update_on_step(self):
        gw: GridWorld = GridWorld(5, 8, num_manipulators=2)
        obs = gw.reset()

        print(gw)
        gw.step([Action.MOVE_DOWN, Action.MOVE_LEFT])
        print(gw)
        gw.step([Action.PICK_UP, Action.PICK_UP])
        print(gw)

    def test_gridworld_environment_should_terminate_after_max_timesteps(self):
        gw: GridWorld = GridWorld(5, 8, num_manipulators=2, max_timesteps=1)
        obs = gw.reset()

        obs, reward, done, info = gw.step([Action.STAY_PUT, Action.STAY_PUT])

        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
