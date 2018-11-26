import torch.nn as nn
import unittest
from ppg.grammar import PolicyGrammar, Token, Goal
from ppg.grammar_test import GrammarTest
from ppg.model import PolicyGrammarNet


class ModelTest(unittest.TestCase):

    def test_model_correctly_parameterizes_grammer(self):
        pg: PolicyGrammar = GrammarTest.make_grammar_small()
        D_agent_state = 10
        D_agent_actions = 5

        def make_activation_net(token: Token) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, 1),
                nn.Sigmoid(),
            )

        def make_production_net(goal: Goal) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, len(goal.productions)),
                nn.Softmax(),
            )

        def make_policy_net(primitive: Primitive) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, D_agent_actions),
                nn.Softmax(),
            )

        net = PolicyGrammarNet(pg, make_activation_net, make_production_net, make_policy_net)

        print(net)


if __name__ == "__main__":
    unittest.main()
