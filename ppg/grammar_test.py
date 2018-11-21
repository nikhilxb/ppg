import unittest
from typing import Mapping
from ppg.grammar import PolicyGrammar, Token, Primitive, Goal


class GrammarTest(unittest.TestCase):

    @staticmethod
    def make_grammar():
        """Construct an example policy grammar."""
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
        pg: PolicyGrammar = GrammarTest.make_grammar()
        print(pg)

        goals: Mapping[str, Goal] = pg.get_goals()
        self.assertEqual(len(goals), 4)

        primitives: Mapping[str, Primitive] = pg.get_primitives()
        self.assertEqual(len(primitives), 8)

        productions: Mapping[str, Token] = pg.get_productions()
        self.assertEqual(len(productions), 4)
        for goal, num in {"AddEgg": 2, "AddFlour": 2, "MakeDough": 4, "MakeCookies": 3}.items():
            self.assertEqual(len(productions[goal]), num)

    def test_grammar_computes_correct_forward(self):
        pg: PolicyGrammar = GrammarTest.make_grammar()
        self.assertTrue(True)  # TODO


if __name__ == "__main__":
    unittest.main()
