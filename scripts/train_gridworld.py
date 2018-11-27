import argparse
import pendulum
import torch
import torch.nn as nn
from ppg.grammar import PolicyGrammar
from ppg.model import PolicyGrammarNet
from worlds.gridworld import Item, GridWorld

# ==================================================================================================
# Command-line arguments.

parser = argparse.ArgumentParser()

# Title.
# ------

parser.add_argument(
    "--experiment_name",
    default="untitled-{}".format(pendulum.now("America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S")),
)

parser.add_argument("--arg_name", type=int, default=1)

# ==================================================================================================
# Policy grammar definition.

pg = PolicyGrammar(
    primitives=[
        "UseToolshed",
        "UseWorkbench",
        "UseFactory",
        "GetIron",
        "GetGrass",
        "GetWood",
    ],
    goals=[
        "MakePlank",
        "MakeStick",
        "MakeCloth",
        "MakeRope",
        "MakeBridge",
        "MakeShears",
        "MakeAxe",
        "MakeBed",
        "MakeLadder",
    ],
)

pi, g = pg.get_tokens()
pg.add_productions("MakePlank", [pi["GetWood"], pi["UseToolshed"]])
pg.add_productions("MakeStick", [pi["GetWood"], pi["UseWorkbench"]])
pg.add_productions("MakeCloth", [pi["GetGrass"], pi["UseFactory"]])
pg.add_productions("MakeRope", [pi["GetGrass"], pi["UseToolshed"]])
pg.add_productions("MakeBridge", [pi["GetIron"], pi["GetWood"], pi["UseFactory"]])
pg.add_productions("MakeShears", [g["MakeStick"], pi["UseToolshed"]])
pg.add_productions("MakeAxe", [g["MakeStick"], pi["GetIron"], pi["UseToolshed"]])
pg.add_productions("MakeBed", [g["MakePlank"], pi["GetGrass"], pi["UseWorkbench"]])
pg.add_productions("MakeLadder", [g["MakePlank"], g["MakeStick"], pi["UseFactory"]])
print(pg)

# ==================================================================================================
# Training functions.

class GridWorldModel(nn.Module):

    def __init__(
            self,
            grammar: PolicyGrammar,
            tasks: List[str],
    ):
        self.grammar: PolicyGrammar = grammar
        self.tasks: List[str] = tasks

        def make_activation_net(token: Token) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
                nn.Sigmoid(),
            )

        def make_production_net(goal: Goal) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, 32),
                nn.ReLU(),
                nn.Linear(32, len(goal.productions))
                nn.Softmax(),
            )

        def make_policy_net(primitive: Primitive) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, 32),
                nn.ReLU(),
                nn.Linear(32, D_agent_actions)
                nn.Softmax(),
            )

        self.policy_net = PolicyGrammarNet(
            pg, make_activation_net, make_production_net, make_policy_net
        )
        self.critic_nets = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(D_agent_state, 1),
            ) for task in self.tasks
        })
        self.state_net = nn.Sequential()  # RNN

        # TODO: state net RNN
        # Implement training step and loop

    def forward(self):
        pass


def train_step(curriculum, max_num_rollouts=100):
    # Generate dataset of rollouts given current curriculum over tasks.
    dataset = []
    while len(dataset) < max_num_rollouts:
        task = sample_task(curriculum)
        rollout = do_rollout(task, model, ...)
        dataset.append(rollout)
    for rollout in dataset:
        model.update_policy()
        model.update_critic()


def train_loop(tasks, max_task_complexity=50, good_task_reward=0.7):
    task_complexity = 1
    while task_complexity < max_task_complexity:
        min_task_reward = float("-inf")
        active_tasks = [t for t in tasks if t.complexity > task_complexity]
        curriculum = torch.uniform(len(active_tasks))
        while min_task_reward < good_task_reward:
            results = train_step(curriculum)
            curriculum = update_curriculum(results)
            min_task_reward = min([mean(task.reward) for task in results])
        task_complexity += 1

# From Andreas et al.
NUM_ROWS = 10
NUM_COLS = 10

def main(args):
    tasks: Mapping = {
        "MakePlank": (GridWorld(NUM_ROWS, NUM_COLS, Item.PLANK)),
        "MakeStick": (GridWorld(NUM_ROWS, NUM_COLS, Item.STICK)),
        "MakeCloth": (GridWorld(NUM_ROWS, NUM_COLS, Item.CLOTH)),
        "MakeRope":  (GridWorld(NUM_ROWS, NUM_COLS, Item.ROPE)),
        "MakeBridge": (GridWorld(NUM_ROWS, NUM_COLS, Item.BRIDGE)),
        "MakeShears": (GridWorld(NUM_ROWS, NUM_COLS, Item.SHEARS)),
    }
    model = GridWorldModel(pg, tasks)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
