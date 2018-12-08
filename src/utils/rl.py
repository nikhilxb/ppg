import torch
import torch.nn as nn
from typing import Sequence, List
from utils.data import Struct


class Sample(Struct):
    """Stored data from a single environment interaction.

    Fields can have any name, although common quantities should use the following canonical
    names that the utility functions assume:
        obs:          Environment observation, Tensor.
        state:        Agent state, Tensor.
        action:       Agent action, Tensor.
        reward:       Environment reward, Tensor.
        ret:          Discounted return, Tensor.
        log_prob:     Log probability of a specific agent action, Tensor.
        state_value:  Expected return from agent state, Tensor.
        action_value: Expected return from agent state taking action, Tensor.
        entropy:      Entropy of entire action distribution, Tensor.
        advantage:    Advantage of return w.r.t. state value, Tensor.
    """

    def __repr__(self) -> str:
        return "Sample {}".format(super().__repr__())

    def __str__(self) -> str:
        return self.__repr__()


class Trajectory(Struct):
    """Aggregated of stored data from many samples.

    All samples are expected to have the same fields. Tensors will be batched along dimension 0.
    Python elements will be batched in a list.
    """

    def __init__(self, samples: Sequence[Sample], collate_fn=None):

        def default_collate(elems: List):
            if isinstance(elems[0], torch.Tensor):
                return torch.stack(elems)
            else:
                return elems

        if collate_fn is None: collate_fn = default_collate

        for k in samples[0].keys():
            self.update(**{k: collate_fn([getattr(s, k) for s in samples])})

    def __repr__(self) -> str:
        return "Trajectory {}".format(super().__repr__())

    def __str__(self) -> str:
        return self.__repr__()


def compute_returns(
        samples: Sequence[Sample],
        discount: float = 1.0,
        device="cpu",
) -> Sequence[Sample]:
    """Fills in `ret` fields in-place for each `Sample`."""
    curr_return = torch.tensor(0.0).to(device)
    for t in reversed(range(len(samples))):
        curr_return = samples[t].reward + discount * curr_return
        samples[t].update(ret=curr_return)
    return samples


def compute_advantages(samples: Sequence[Sample]) -> Sequence[Sample]:
    """Fill in `advantage` fields in-place for each `Sample`."""
    for t in range(len(samples)):
        advantage = samples[t].ret - samples[t].state_value
        samples[t].update(advantage=advantage)
    return samples


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
