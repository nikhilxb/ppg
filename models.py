import torch
import torch.nn as nn
from ppg.grammar import PolicyGrammar, Token, Goal, Primitive, PolicyGrammarNet


class GridWorldAgent(nn.Module):

    def __init__(
            self,
            grammar: PolicyGrammar,
            agent_state_dim: int = 100,
            agent_action_dim: int = 5,
            activation_net_hidden_dim: int = 32,
            production_net_hidden_dim: int = 32,
            policy_net_hidden_dim: int = 32,
            state_net_hidden_dim: int = 32,
    ):
        self.grammar: PolicyGrammar = grammar

        # Initial hidden state shape: (num_layers, batch, hidden dimension)
        # TODO: What is the batch size? set to 1 for now
        self.state_net_layers = 1
        self.state_net_batch = 1
        self.state_net_hidden_dim = state_net_hidden_dim
        self.hidden_state = torch.new_zeros((self.state_net_layers, self.state_net_batch, self.state_net_hidden_dim))

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

        self.policy_net = PolicyGrammarNet(
            self.grammar, make_activation_net, make_production_net, make_policy_net
        )
        self.state_net = nn.GRU(agent_state_dim, state_net_hidden_dim)

    def reset(self):
        # Reset the agent's state net
        self.hidden_state = torch.new_zeros((self.state_net_layers, self.state_net_batch, self.state_net_hidden_dim))

    def forward(
            self,
            observation: torch.Tensor,
            goal: str,
    ):
        agent_state, next_hidden_state = self.state_net(observation, self.hidden_state)
        self.hidden_state = next_hidden_state
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
            policy_net_hidden_dim: int = 32,
            agent_action_dim: int = 6,
    ):
    # NOTE: the agent action dimension is now defaulted to 6, because
    # Andreas et al augument their action space with a STOP token
