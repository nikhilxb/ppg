from typing import List, Tuple, Dict
import torch

"""
Policy primitive outputs a softmax over action space
N_actions --> dimensionality of action space

probs equiv log(probs)
"""

class PolicyGrammar(object):
    """This class specifies a probabilistic policy grammar."""

    def __init__(self):
        self.rules: Dict[Goal, List[Production]] = {}  # {"subgoal": }

    def forward(self, root: Token) -> Tensor[N_actions]:
        """
        Forward pass for the PPG DAG; construct the probability distribution
        P(a | s)
        """
        # Base case: Policy primitives return probs over actions.
        if(root.is_primitive): return root.policy_probs

        # Recursive case:
        for production in self.rules[root]:
            # How do we work production and activation probabilities into this?
            return self.forward

    """
    Testing pushing code w/atom! Also, I think I've come up with a recurrence
    relation we can use, although I'm not quite sure how to work the "soft"
    Boolean logic into the formulation yet
    """

class Token(object):
    def __init__(self,
                 activation_prob: Tensor,
                 is_primitive: bool):
        self.activation_prob: Tensor = activation_prob
        self.is_primitive: bool = is_primitive

class Goal(Token):
    def __init__(self,
                 name: str):
        self.name: str = name

class PolicyPrimitive(Token):
    def __init__(self,
                 name: str,
                 action_probs: Tensor):
        pass

class Production(object):
    def __init__(self,
                 tokens: List[Token],
                 production_prob: Tensor):
        pass
