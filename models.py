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
        self.state_net = nn.Sequential(nn.Linear(agent_state_dim, state_net_hidden_dim),
                                      )  # TODO: Make into RNN

    def reset_hidden(self):

    def forward(
            self,
            observation: torch.Tensor,
            goal: str,
    ):
        self.hidden_state
        agent_state = self.state_net(observation)
        action_scores = self.policy_net(goal, agent_state)
        return agent_state, action_scores
