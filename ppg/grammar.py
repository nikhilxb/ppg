# -*- coding: utf-8 -*-
import crayons
from typing import List, Tuple, Mapping, Callable, Union, Any, Optional, cast
from collections import defaultdict


class Token:
    """This class defines an abstract token, which can be either a goal or policy primitive."""

    def __init__(
            self,
            name: str,
            is_primitive: bool,
            activation_prob: Callable = lambda state: 0,
    ):
        self.name = name
        self.is_primitive: bool = is_primitive
        self.activation_prob: Callable = activation_prob

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
            activation_prob: Callable = lambda state: 0,
            production_probs: Callable = lambda state: [0],
            productions: Optional[List[Token]] = None,
    ):
        super().__init__(name, False, activation_prob=activation_prob)
        self.production_probs: Callable = production_probs
        self.productions: List[Token] = [] if productions is None else productions


class Primitive(Token):
    """This class defines a policy primitive, which is a terminal token in the policy grammar.

    A policy primitive has an additional `policy_probs` property, which defines the probabilities of
    taking each possible agent action under the policy.
    """

    def __init__(
            self,
            name: str,
            activation_prob: Callable = lambda state: 0,
            policy_probs: Callable = lambda state: [0],
    ):
        super().__init__(name, True, activation_prob=activation_prob)
        self.policy_probs: Callable = policy_probs


class PolicyGrammar:
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
        for pi in primitives:
            self.add_primitive(pi)
        for g in goals:
            self.add_goal(g)

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
            raise ValueError(
                "Token specification must be of type 'str' or '{}';"
                "got {} (of type {})".format(token_cls.__name__, token, type(token))
            )

        # Ensure registered token is unique
        if name in store:
            raise ValueError("{} '{}' already defined".format(token_cls.__name__, name))
        else:
            store[name] = obj

    def add_primitive(self, primitive: Union[str, Primitive]) -> None:
        PolicyGrammar._add_token(self.primitives, primitive, Primitive)

    def add_goal(self, goal: Union[str, Goal]) -> None:
        PolicyGrammar._add_token(self.goals, goal, Goal)

    def add_production(self, goal: str, production: Token) -> None:
        if goal not in self.goals:
            raise ValueError("Invalid goal name in rule; got {}".format(goal))
        self.goals[goal].productions.append(production)

    def add_productions(self, goal: str, productions: List[Token]) -> None:
        for production in productions:
            self.add_production(goal, production)

    def get_primitives(self) -> Mapping[str, Primitive]:
        return self.primitives

    def get_goals(self) -> Mapping[str, Goal]:
        return self.goals

    def get_tokens(self) -> Tuple[Mapping[str, Primitive], Mapping[str, Goal]]:
        return self.primitives, self.goals

    def get_productions(self) -> Mapping[str, List[Token]]:
        return {name: goal.productions for name, goal in self.goals.items()}

    def __str__(self) -> str:

        def color(s, is_primitive):
            return crayons.red(s) if is_primitive else crayons.blue(s)

        prims = "\n".join("    {}".format(color(pi, True)) for pi in self.primitives)
        goals = "\n".join("    {}".format(color(g, False)) for g in self.goals)
        rules = "\n".join(
            "    {} --> {}".format(
                color(gname, False),
                " | ".join([str(color(prod, prod.is_primitive)) for prod in gobj.productions]),
            ) for gname, gobj in self.goals.items()
        )
        return "Primitives:\n{}\nGoals:\n{}\nRules:\n{}".format(prims, goals, rules)

    def forward(self, start_goal: str, agent_state: Any) -> Any:
        """Construct probability over each action by expanding the policy grammar starting from
        the specified top-level goal.
        :param start_goal
        :param agent_state
        :return
        """
        cache: Mapping[Token, Any] = {}

        def recurse(root: Token) -> Any:
            # Cache case: Prevent re-computation
            if root in cache:
                return cache[root]

            # Base case: Policy primitives return probs over action
            if root.is_primitive:
                root = cast(Primitive, root)
                return root.activation_prob(agent_state) * root.policy_probs(agent_state)

            # Recursive case: Traverse all possible productions from goal.
            root = cast(Goal, root)
            downstream_probs = 0
            production_probs = root.production_probs(agent_state)
            for i, production in enumerate(root.productions):
                downstream_probs += production_probs[i] * recurse(production)
            return root.activation_prob(agent_state) * downstream_probs

        return recurse(self.goals[start_goal])
