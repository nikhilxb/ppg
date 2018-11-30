import unittest
import torch
from worlds.gridworld import GridWorld, Item
from scripts.train_gridworld import encode_observation, calc_observation_dim


class TrainGridWorldTest(unittest.TestCase):

    def test_encode_observation_returns_correct_tensor(self):
        window_radius: int = 2
        gw: GridWorld = GridWorld(5, 8, Item.PLANK, window_radius=window_radius)
        obs = gw.reset()

        encoded_obs: torch.Tensor = encode_observation(obs)

        self.assertEqual(encoded_obs.size()[0], calc_observation_dim(window_radius))


if __name__ == "__main__":
    unittest.main()
