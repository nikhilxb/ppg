from typing import List, Tuple, Dict
import torch

"""
An alternate implementation of the PPG. In this case, the transition (production)
probabilities are included in the Token class, instead of a seperate Production
class. Also, I'm assuming the list of production rules simply gives the
nodes that can be reached from the initial Goal node by following one edge. That
is, if g_0 --> g_1 | g_2 | g_3, we have Goal = g_0 and
List[Production] = [g_1, g_2, g_3]. Each element of the list is assumed to be
a Goal or PolicyPrimitive object

"""

class PolicyGrammar(object):
    """This class specifies a probabilistic policy grammar."""

    def __init__(self):
        self.rules: Dict[Goal, List[Production]] = {}

    def forward(self, root: Token) -> Tensor[N_actions]:
        """
        Forward pass for the PPG DAG; construct the probability distribution
        P(a | s)
        """
        # Base case: Policy primitives return probs over action
        if(root.is_primitive): return root.action_probs*root.activation_prob

        # Recursive case
        """
        In the recursive case, i is the index of the transition [root --> production]
        in the vector of normalized probabilities returned by the root neural network.
        Not sure how we'll get that yet since we haven't coded up the networks,
        but I wanted to get the basic recursion relation down.
        """
        prob = 0
        for production in self.rules[root]:
            """ ** Won't Compile ** (see comments above) """
            prob += root.activation_prob*root.transition_prob[i]*self.forward(production)

        return prob

class Token(object):
    def __init__(self,
                 activation_prob: Tensor,
                 is_primitive: bool):
        self.activation_prob: Tensor = activation_prob
        self.is_primitive: bool = is_primitive

class Goal(Token):
    def __init__(self,
                 name: str,
                 transition_prob: Tensor,):
        self.name: str = name
        self.transition_prob: Tensor = transition_prob

class PolicyPrimitive(Token):
    def __init__(self,
                 name: str,
                 action_probs: Tensor):
        self.action_probs: Tensor = action_probs
