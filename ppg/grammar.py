# -*- coding: utf-8 -*-
from typing import List, Tuple, Mapping, Union, Optional
from collections import defaultdict
import torch
import crayons


class Token(object):
    """This class defines an abstract token, which can be either a goal or policy primitive."""
    def __init__(
        self,
        name: str,
        is_primitive: bool,
        activation_prob: Optional[torch.Tensor] = None,
    ):
        self.name = name
        self.is_primitive: bool = is_primitive
        self.activation_prob: Optional[torch.Tensor] = activation_prob

    def __str__(self):
        return self.name


class Goal(Token):
    """This class defines a goal, which is a non-terminal token in the policy grammar.

    A goal token has an additional `production_probs` property, which defines the probabilities of
    producing each of the productions defined for the goal.
    """
    def __init__(
        self,
        name: str,
        activation_prob: Optional[torch.Tensor] = None,
        production_probs: Optional[torch.Tensor] = None,
    ):
        super().__init__(name, False, activation_prob=activation_prob)
        self.production_probs: Optional[torch.Tensor] = production_probs


class Primitive(Token):
    """This class defines a policy primitive, which is a terminal token in the policy grammar.

    A policy primitive has an additional `action_probs` property, which defines the probabilities of
    taking each possible agent action under the policy.
    """
    def __init__(
        self,
        name: str,
        activation_prob: Optional[torch.Tensor] = None,
        action_probs: Optional[torch.Tensor] = None,
    ):
        super().__init__(name, True, activation_prob=activation_prob)
        self.action_probs: Optional[torch.Tensor] = action_probs


class PolicyGrammar(object):
    """This class specifies a probabilistic policy grammar, which decomposes the choice of an agent
    action using a hierarchy of goals and policy primitives.

    Formally, the policy grammar is defined by:

        `s ∈ S
            Agent states.
        `a ∈ A`
            Agent actions.
        `g_n ∈ G`
            Goal tokens (non-terminal). There are `1 <= n <= N` goal tokens in total.
        `π_k ∈ Π`
            Policy primitives (terminal), where π_k(a | s) = Prob[a | s]. There are `1 <= k <= K`
            policy primitives in total.
        `g_n → z_n_i`
            Production rules, where `z_n_i ∈ (G ∪ Π)` can be a goal token or a policy primitive.
            There are different numbers `1 <= i <= I(n)` of productions per goal.
        `activation_prob(g_n | s)`
        `activation_prob(π_k | s)`
            Probabilities of each goal and policy primitive being activated, given state `s`.
        `production_prob(z_n_i | g_n, s)`
            Probabilities of each production for a goal, given state `s`.
            The sum over all `z_n_i` for a goal `g_n` is 1.

    The probability of an agent action `a` given the state `s` and a top-level goal `g_0` is:

        ```
        Prob[a, G, Π | s] = f(g_0)
        f(z) = {
            if z is a policy primitive π:
                activation_prob(π | s) * π(a | s),
            elif z is a goal token g:
                activation_prob(g | s) * \sum{z'} production_prob(z' | g, s) * f(z')
        }
        ```
    """

    def __init__(
        self,
        primitives: List[Union[str, Primitive]] = [],
        goals: List[Union[str, Goal]] = [],
    ):
        # Register all primitives and goals
        self.primitives: Mapping[str, Primitive] = {}
        self.goals: Mapping[str, Goal] = {}
        for pi in primitives: self.add_primitive(pi)
        for g in goals: self.add_goal(g)

        # Map from goal name to list of defined productions
        self.rules: Mapping[str, List[Token]] = defaultdict(list)

    @staticmethod
    def _add_token(store: Mapping[str, Token], token: Union[str, Token], token_cls):
        # Extract name and object from token specification
        name: str
        obj: Token
        if isinstance(token, str):
            name = token
            obj = token_cls(name)
        elif isinstance(token, token_cls):
            obj = token
            name = obj.name
        else:
            raise ValueError("Token specification must be of type 'str' or '{}';"
                             "got {} (of type {})".format(token_cls.__name__, token, type(token)))

        # Ensure registered token is unique
        if name in store:
            raise ValueError("{} '{}' already defined".format(token_cls.__name__, name))
        else:
            store[name] = obj

    def add_primitive(self, primitive: Union[str, Primitive]):
        PolicyGrammar._add_token(self.primitives, primitive, Primitive)

    def add_goal(self, goal: Union[str, Goal]):
        PolicyGrammar._add_token(self.goals, goal, Goal)

    def add_rule(self, goal: str, production: Token):
        if goal not in self.goals:
            raise ValueError("Invalid goal name in rule; got {}".format(goal))
        self.rules[goal].append(production)

    def add_rules(self, goal: str, productions: List[Token]):
        for production in productions:
            self.add_rule(goal, production)

    def get_primitives(self) -> Mapping[str, Primitive]:
        return self.primitives

    def get_goals(self) -> Mapping[str, Goal]:
        return self.goals

    def get_tokens(self) -> Tuple[Mapping[str, Primitive], Mapping[str, Goal]]:
        return self.primitives, self.goals

    def __str__(self):
        prims = "\n".join(["    {}".format(crayons.red(pi)) for pi in self.primitives])
        goals = "\n".join(["    {}".format(crayons.blue(g)) for g in self.goals])
        rules = "\n".join(["    {} --> {}".format(
            crayons.blue(g),
            " | ".join([str(crayons.red(t)) if t.is_primitive else str(crayons.blue(t))
                        for t in tokens]))
            for g, tokens in self.rules.items()]
        )

        return "Primitives:\n{}\nGoals:\n{}\nRules:\n{}".format(prims, goals, rules)

    def forward(self, root: Token) -> torch.Tensor:
        """Construct Prob[a, G, Π | s] starting at goal `g`"""
        # TODO: Make this work

        # Base case: Policy primitives return probs over action
        if root.is_primitive:
            return root.activation_prob * root.action_probs

        # Recursive case: Traverse all possible productions from goal.
        probs: torch.Tensor = 0
        # NOTE: doesn't look like the token class has a "productions" attribute 
        for i, production in enumerate(root.productions):
            probs += root.production_probs[i] * self.forward(production)
        return root.activation_prob * probs


def test_grammar():
    pg = PolicyGrammar(
        primitives=[
            "PickEgg",
            "BreakEgg",
            "PickFlour",
            "PourFlour",
            "MixDough",
            "KneadDough",
            "ShapeCookies",
            "BakeCookies"
        ],
        goals=[
            "AddEgg",
            "AddFlour",
            "MakeDough",
            "MakeCookies"
        ]
    )
    pi, g = pg.get_tokens()

    pg.add_rules("AddEgg", [
        pi["PickEgg"], pi["BreakEgg"],
    ])
    pg.add_rules("AddFlour", [
        pi["PickFlour"], pi["PourFlour"]
    ])
    pg.add_rules("MakeDough", [
        g["AddEgg"], g["AddFlour"], pi["MixDough"], pi["KneadDough"]
    ])
    pg.add_rules("MakeCookies", [
        g["MakeDough"], pi["ShapeCookies"], pi["BakeCookies"]
    ])

    print(pg)


if __name__ == "__main__":
    test_grammar()
