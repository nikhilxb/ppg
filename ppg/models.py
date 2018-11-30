import torch
import torch.nn as nn
from typing import Tuple
from ppg.grammar import PolicyGrammar, Token, Goal, Primitive, PolicyGrammarNet


class GridWorldAgent(nn.Module):

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
        self.reset()

    def reset(self) -> None:
        self.hidden_state = torch.zeros(self.state_net_layers_num, 1, self.agent_state_dim)

    def forward(self, observation: torch.Tensor, goal: str) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_state, self.hidden_state = self.state_net(
            observation.view(1, 1, -1),
            self.hidden_state,
        )
        agent_state = agent_state.view(-1)
        action_scores = self.policy_net(goal, agent_state)
        return agent_state, action_scores


class Baseline(nn.Module):
    """
    Baseline model; an implementation of the model in the Andreas et al Policy Sketches paper.
    This model *only* includes policy primitive networks ("policy_nets" in the above class).
    """

    def __init__(
            self,
            agent_state_dim: int = 100,
            baseline_net_hidden_dim: int = 32,
            agent_action_dim: int = 6,
    ):
        # NOTE: the agent action dimension is now defaulted to 6, because
        # Andreas et al augument their action space with a STOP token
        def make_baseline_net(primitive: Primitive) -> nn.Module:
            return nn.Sequential(
                nn.Linear(agent_state_dim, baseline_net_hidden_dim),
                nn.ReLU(),
                nn.Linear(baseline_net_hidden_dim, agent_action_dim),
                nn.Softmax(),
            )
