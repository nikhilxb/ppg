import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, List, Mapping, Sequence, Optional
from ppg.grammar import PolicyGrammar, Token, Goal, Primitive, PolicyGrammarNet


class PolicyGrammarAgent(nn.Module):

    def __init__(
            self,
            grammar: PolicyGrammar,
            env_observation_dim: int = 10,
            agent_state_dim: int = 100,
            agent_action_dim: int = 5,
            activation_net_hidden_dim: int = 32,
            production_net_hidden_dim: int = 32,
            policy_net_hidden_dim: int = 32,
            state_net_layers_num: int = 1,
    ):
        super().__init__()
        self.grammar: PolicyGrammar = grammar

        def make_activation_net(token: Token) -> nn.Module:
            return nn.Sequential(
                nn.Linear(agent_state_dim, activation_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(activation_net_hidden_dim, 1),
                nn.Sigmoid(),
            )

        def make_production_net(goal: Goal) -> nn.Module:
            return nn.Sequential(
                nn.Linear(agent_state_dim, production_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(production_net_hidden_dim, len(goal.productions)),
                nn.Softmax(),
            )

        def make_policy_net(primitive: Primitive) -> nn.Module:
            return nn.Sequential(
                nn.Linear(agent_state_dim, policy_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(policy_net_hidden_dim, agent_action_dim),
                nn.Softmax(),
            )

        # Policy net converts agent state into policy action scores.
        self.policy_net = PolicyGrammarNet(
            self.grammar, make_activation_net, make_production_net, make_policy_net
        )

        # State net integrates environment observation into agent recurrent state,
        # of size (num_layers, batch=1, agent_state_dim).
        self.agent_state_dim: int = agent_state_dim
        self.state_net_layers_num: int = state_net_layers_num
        self.state_net = nn.GRU(
            env_observation_dim, agent_state_dim, num_layers=state_net_layers_num
        )
        self.hidden_state = None
        self.current_goal = None

    def reset(self, goal: str, device="cpu") -> None:
        """
        :param goal:
            Name of top-level `Goal` in `PolicyGrammar`.
        :param device:
        """
        self.current_goal = goal
        self.hidden_state = torch.zeros(
            (self.state_net_layers_num, 1, self.agent_state_dim), device=device
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param obs:
            Environment observation, size[timesteps, env_observation_dim].
        :return:
            Agent state, size[timesteps, agent_state_dim].
            Action probabilities, size[timesteps, agent_action_dim].
        """
        agent_state, self.hidden_state = self.state_net(
            obs.unsqueeze(1),
            self.hidden_state,
        )
        agent_state = agent_state.squeeze(1)
        action_probs = self.policy_net(self.current_goal, agent_state)
        return agent_state, action_probs


class PolicySketchAgent(nn.Module):
    """
    Baseline model; an implementation of the model in the Andreas et al Policy Sketches paper.
    This model *only* includes policy primitive networks ("policy_nets" in the above class).
    """

    def __init__(
            self,
            sketches: Mapping[str, Sequence[str]],
            env_observation_dim: int = 10,
            agent_state_dim: int = 100,
            agent_action_dim: int = 5,
            policy_net_hidden_dim: int = 32,
            state_net_layers_num: int = 1,
    ):
        super().__init__()

        def make_baseline_net() -> nn.Module:
            return nn.Sequential(
                nn.Linear(agent_state_dim, policy_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(policy_net_hidden_dim, agent_action_dim + 1),
                nn.Softmax(),
            )

        self.sketches: Mapping[str, Sequence[str]] = sketches
        self.primitives: Mapping[str, nn.Module] = {}
        for goal, primitives in sketches.items():
            for primitive in primitives:
                if primitive in self.primitives: continue
                self.primitives[primitive] = make_baseline_net()

        self.current_goal = None
        self.current_primitive = None
        self.current_primitive_idx = None

        self.agent_state_dim: int = agent_state_dim
        self.state_net_layers_num = state_net_layers_num

        self.state_net = nn.GRU(
            env_observation_dim, agent_state_dim, num_layers=state_net_layers_num
        )
        self.hidden_state = None

    def reset(self, goal: str, device="cpu"):
        self.current_goal = goal
        self.current_primitive_idx = 0
        self.hidden_state = torch.zeros(
            (self.state_net_layers_num, 1, self.agent_state_dim), device=device
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        :param obs:
            Environment observation, size[1, env_observation_dim].
        :return:
            Agent state, size[1, agent_state_dim].
            Action probabilities, size[1, agent_action_dim].
            Primitive index used.
        """
        agent_state, self.hidden_state = self.state_net(
            obs.unsqueeze(1),
            self.hidden_state,
        )
        agent_state = agent_state.squeeze(1)  # size, [1, agent_state_dim]
        curr_sketch: List[str] = self.sketches[self.current_goal]

        action_probs = self.primitives[curr_sketch[self.current_primitive_idx]](agent_state)
        action = dist.Categorical(action_probs.squeeze(0)).sample().item()

        # If sampled STOP and is valid to advance, recompute action scores using next primitive.
        if action == 0 and self.current_primitive_idx + 1 < len(curr_sketch):
            self.current_primitive_idx += 1
            action_probs = self.primitives[curr_sketch[self.current_primitive_idx]](agent_state)

        return agent_state, action_probs[:, 1:], self.current_primitive_idx

    def evaluate(self, obs: torch.Tensor,
                 primitive_idxs: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param obs:
            Environment observation, size[timesteps, env_observation_dim].
        :param primitive_idxs:
            Primitive indices to use (corresponds to a unique network, List[timesteps].
        :return:
            Agent state, size[timesteps, agent_state_dim].
            Action probabilities, size[timesteps, agent_action_dim].
        """
        agent_state, self.hidden_state = self.state_net(
            obs.unsqueeze(1),
            self.hidden_state,
        )
        agent_state = agent_state.squeeze(1)  # size, [timesteps, agent_state_dim]
        curr_sketch: List[str] = self.sketches[self.current_goal]

        action_probs = torch.stack(
            [
                self.primitives[curr_sketch[idx]](agent_state[t, :])
                for t, idx in enumerate(primitive_idxs)
            ]
        )  # size[timesteps, agent_action_dim]

        return agent_state, action_probs[:, 1:]
