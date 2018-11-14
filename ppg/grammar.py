from typing import List, Tuple, Dict
import torch


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

    def __init__(self):
        self.rules: Dict[Goal, List[Production]] = {}

    def forward(self, root: Token) -> torch.Tensor:
        """Construct Prob[a, G, Π | s] starting at goal `g`"""

        # Base case: Policy primitives return probs over action
        if root.is_primitive: return root.activation_prob * root.action_probs

        # Recursive case: Traverse all possible productions from goal.
        probs: torch.Tensor = 0
        for i, production in enumerate(root.productions):
            probs += root.production_probs[i] * self.forward(production)
        return root.activation_prob * probs


class Token(object):
    def __init__(
        self,
        activation_prob: torch.Tensor,
        is_primitive: bool
    ):
        self.activation_prob: torch.Tensor = activation_prob
        self.is_primitive: bool = is_primitive


class Goal(Token):
    def __init__(
        self,
        name: str,
        activation_prob: torch.Tensor,
        production_probs: torch.Tensor,
    ):
        super().__init__(activation_prob, is_primitive=False)
        self.name: str = name
        self.production_probs: torch.Tensor = production_probs


class PolicyPrimitive(Token):
    def __init__(
        self,
        name: str,
        activation_prob: torch.Tensor,
        action_probs: torch.Tensor
    ):
        super().__init__(activation_prob, is_primitive=True)
        self.name: str = name
        self.action_probs: torch.Tensor = action_probs
