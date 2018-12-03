import unittest
import numpy as np
import torch.nn as nn
from typing import Mapping
from ppg.grammar import PolicyGrammar, Token, Primitive, Goal, PolicyGrammarNet


class GrammarTest(unittest.TestCase):

    @staticmethod
    def make_grammar_small():
        """Construct an small example policy grammar."""
        pg = PolicyGrammar(
            primitives=[
                "Prim1",
                "Prim2",
                "Prim3",
            ],
            goals=[
                "Goal1",
                "Goal2",
            ],
        )
        pi, g = pg.get_tokens()
        pg.add_productions(
            "Goal1",
            [
                g["Goal2"],
                pi["Prim3"],
            ],
        )
        pg.add_productions(
            "Goal2",
            [
                pi["Prim1"],
                pi["Prim2"],
            ],
        )

        return pg

    @staticmethod
    def make_grammar_large():
        """Construct a large example policy grammar."""
        pg = PolicyGrammar(
            primitives=[
                "PickEgg",
                "BreakEgg",
                "PickFlour",
                "PourFlour",
                "MixDough",
                "KneadDough",
                "ShapeCookies",
                "BakeCookies",
            ],
            goals=[
                "AddEgg",
                "AddFlour",
                "MakeDough",
                "MakeCookies",
            ],
        )
        pi, g = pg.get_tokens()
        pg.add_productions(
            "AddEgg",
            [
                pi["PickEgg"],
                pi["BreakEgg"],
            ],
        )
        pg.add_productions(
            "AddFlour",
            [
                pi["PickFlour"],
                pi["PourFlour"],
            ],
        )
        pg.add_productions(
            "MakeDough",
            [
                g["AddEgg"],
                g["AddFlour"],
                pi["MixDough"],
                pi["KneadDough"],
            ],
        )
        pg.add_productions(
            "MakeCookies",
            [
                g["MakeDough"],
                pi["ShapeCookies"],
                pi["BakeCookies"],
            ],
        )
        return pg

    def test_grammar_has_correct_number_of_tokens_and_productions(self):
        pg: PolicyGrammar = GrammarTest.make_grammar_large()
        print(pg)

        goals: Mapping[str, Goal] = pg.get_goals()
        self.assertEqual(len(goals), 4)

        primitives: Mapping[str, Primitive] = pg.get_primitives()
        self.assertEqual(len(primitives), 8)

        productions: Mapping[str, Token] = pg.get_productions()
        self.assertEqual(len(productions), 4)
        for goal, num in {"AddEgg": 2, "AddFlour": 2, "MakeDough": 4, "MakeCookies": 3}.items():
            self.assertEqual(len(productions[goal]), num)

    def test_grammar_computes_forward(self):
        pg: PolicyGrammar = GrammarTest.make_grammar_small()
        print(pg)

        pi, g = pg.get_tokens()

        pi["Prim1"].activation_prob = lambda state: 1.0
        pi["Prim1"].policy_probs = lambda state: np.array([0.5, 0.5])

        pi["Prim2"].activation_prob = lambda state: 1.0
        pi["Prim2"].policy_probs = lambda state: np.array([0.7, 0.3])

        pi["Prim3"].activation_prob = lambda state: 0.8
        pi["Prim3"].policy_probs = lambda state: np.array([0.1, 0.9])

        g["Goal1"].activation_prob = lambda state: 1.0
        g["Goal1"].production_probs = lambda state: np.array([0.8, 0.2])

        g["Goal2"].activation_prob = lambda state: 1.0
        g["Goal2"].production_probs = lambda state: np.array([0.6, 0.4])

        policy_weights = pg.forward("Goal1", None)
        self.assertAlmostEqual(policy_weights[0], 0.48)
        self.assertAlmostEqual(policy_weights[1], 0.48)

    def test_grammar_computed_forward_with_conditioning(self):
        pg: PolicyGrammar = GrammarTest.make_grammar_large()
        self.assertTrue(True)  # TODO

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
