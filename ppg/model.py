# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Callable, Mapping, List
from ppg.grammar import PolicyGrammar, Goal, Token


class PolicyGrammarNet(nn.Module):
    """This module assembles a multi-branch neural network to parametrize the probabilities used
    in the `PolicyGrammar` calculation."""

    def __init__(
            self,
            grammar: PolicyGrammar,
            make_activation_net: Callable[[Token], nn.Module],
            make_production_net: Callable[[Goal], nn.Module],
    ):
        """Constructor.
        :param grammar
            A policy grammar to parametrize.
        :param make_activation_net
            Function to generate a network that transforms agent state to an activation probability:
            Input:
                `Tensor[D_agent_state]`
                    Agent state.
            Output:
                `Tensor[1]`
                    Activation probability.
        :param make_production_net
            Function to generate a network that transforms agent state to production probabilities:
            Input:
                `Tensor[D_agent_state]`
                    Agent state.
            Output:
                `Tensor[N_productions]`
                    Production probabilities for each production. Each goal has a different number
                    of productions.
        """
        super().__init__()
        self.grammar: PolicyGrammar = grammar

        # Construct networks for primitives
        self.activation_nets_primitives: Mapping[str, nn.Module] = nn.ModuleDict()
        for name, primitive in self.grammar.get_primitives().items():
            activation_net: nn.Module = make_activation_net(primitive)
            self.activation_nets_primitives[name] = activation_net
            primitive.activation_prob = lambda state: activation_net(state)

        # Construct networks for goals
        self.activation_nets_goals: Mapping[str, nn.Module] = nn.ModuleDict()
        self.production_nets_goals: Mapping[str, nn.Module] = nn.ModuleDict()
        for name, goal in self.grammar.get_goals().items():
            activation_net: nn.Module = make_activation_net(goal)
            production_net: nn.Module = make_production_net(goal)
            self.activation_nets_goals[name] = activation_net
            self.production_nets_goals[name] = production_net
            goal.activation_prob = lambda state: activation_net(state)
            goal.production_probs = lambda state: production_net(state)

    def forward(self, goal: str, agent_state: torch.Tensor) -> torch.Tensor:
        """Compute the action probabilities using the policy grammar."""
        return self.grammar.forward(goal, agent_state)
