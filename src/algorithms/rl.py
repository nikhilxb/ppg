import torch
import torch.nn as nn
import torch.utils.data.dataset as dataset
from dataclasses import dataclass
from typing import Sequence, Optional, Any


@dataclass
class Transition:
    """Experience gained from a single environment interaction."""
    # yapf: disable
    obs:               Optional[torch.Tensor] = None
    state:             Optional[torch.Tensor] = None
    action:            Optional[torch.Tensor] = None
    reward:            Optional[torch.Tensor] = None
    discounted_return: Optional[torch.Tensor] = None
    log_prob:          Optional[torch.Tensor] = None
    state_value:       Optional[torch.Tensor] = None
    action_value:      Optional[torch.Tensor] = None
    entropy:           Optional[torch.Tensor] = None
    advantage:         Optional[torch.Tensor] = None
    # yapf: enable

    def update(self, **kwargs) -> 'Transition':
        self.__dict__.update(kwargs)
        return self


# Experience gained from an entire episode of environment interactions.
Rollout = Sequence[Transition]

# Experience gained from an entire episode of environment interactions, but vectorized.
VectorizedRollout = Transition


def compute_discounted_returns(rollout: Rollout, discount: float = 1.0, device="cpu") -> Rollout:
    """Fills in `discounted_return` fields in-place for each `Transition` in the `Rollout`."""
    curr_return = torch.tensor([0.0]).to(device)
    for t in reversed(range(len(rollout))):
        curr_return = rollout[t].reward + discount * curr_return
        rollout[t].update(discounted_return=curr_return)
    return rollout


def compute_advantages(rollout: Rollout):
    """Fill in `advantage` fields in-place for each `Transition` in the `Rollout`."""
    for t in range(len(rollout)):
        advantage = rollout[t].discounted_return - rollout[t].state_value
        rollout[t].update(advantage=advantage)
    return rollout


class SequenceDataset(dataset.Dataset):
    """Defines a dataset from a sequence of examples."""

    def __init__(self, examples: Sequence):
        self.examples: Sequence = examples

    def __getitem__(self, i: int) -> Any:
        return self.examples[i]

    def __len__(self) -> int:
        return len(self.examples)


class PPOClipLoss(nn.Module):
    """Creates a criterion that computes the PPO loss as described in (Schulman et al. 2017)."""

    def __init__(
            self,
            clip_epsilon: float = 0.2,
            value_loss_coeff: float = 1.0,
            entropy_loss_coeff: float = 0.1,
    ):
        super().__init__()
        self.clip_epsilon: float = clip_epsilon
        self.value_loss_coeff: float = value_loss_coeff
        self.entropy_loss_coeff: float = entropy_loss_coeff

    def forward(
            self,
            log_prob,
            log_prob_old,
            advantage_old,
            state_value,
            discounted_return,
            entropy,
    ) -> torch.Tensor:
        """
        :param log_prob:
            Discrete action log probability, size[T].
        :param log_prob_old:
            Discrete action log probability for previous version of model parameters, size[T].
        :param advantage_old:
            Action advantage for previous version of model parameters, size[T].
        :param state_value:
            State value prediction, size[T].
        :param discounted_return:
            Discounted return, size[T].
        :param entropy:
            Policy distribution entropy, size[T].
        :return:
            Loss that maximizes policy loss and entropy, and minimizes state value loss, size[1].
        """
        ratios = torch.exp(log_prob - log_prob_old)
        ratios_clamp = ratios.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss = -(torch.min(ratios, ratios_clamp) * advantage_old).mean()
        value_loss = 0.5 * (state_value - discounted_return).pow(2).mean()
        entropy_loss = -entropy.mean()

        return (
            policy_loss + value_loss * self.value_loss_coeff +
            entropy_loss * self.entropy_loss_coeff
        )
