import argparse
import pendulum
import torch
import torch.distributions as dist
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

args = parser.parse_args()

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
            tasks: List[Task],
    ):
        self.grammar: PolicyGrammar = grammar
        self.tasks: List[Task] = tasks

        def make_activation_net(token: Token) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, args.dim_activation_hidden),
                nn.ReLU(),
                nn.Linear(args.dim_activation_hidden, 1),
                nn.Sigmoid(),
            )

        def make_production_net(goal: Goal) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, args.dim_production_hidden),
                nn.ReLU(),
                nn.Linear(args.dim_production_hidden, len(goal.productions)),
                nn.Softmax(),
            )

        def make_policy_net(primitive: Primitive) -> nn.Module:
            return nn.Sequential(
                nn.Linear(D_agent_state, args.dim_policy_hidden),
                nn.ReLU(),
                nn.Linear(args.dim_policy_hidden, D_agent_actions),
                nn.Softmax(),
            )

        self.policy_net = PolicyGrammarNet(
            pg, make_activation_net, make_production_net, make_policy_net
        )
        self.critic_nets = nn.ModuleList(
            [nn.Sequential(nn.Linear(D_agent_state, args.dim_critic_hidden), nn.ReLU(), nn.Linear(args.dim_critic_hidden, 1),) for task in self.tasks]
        )
        self.state_net = nn.Sequential(
            nn.Linear()
        )  # TODO: Make into RNN

        # TODO: state net RNN
        # Implement training step and loop

    def forward(self, observation):
        state = self.state_net(observation)


Rollout = namedtuple("Rollout", ["task_idx", "trajectory"])
# task_idx: int
# trajectory: List[Sample]
Sample = namedtuple("Sample", ["state", "action", "reward"]) # Tuple[torch.Tensor, Action, float]

def do_rollout(
        task: Task,
        model: GridWorldModel
):
    # Use the algorithm to rollout a trajectory, given the task
    pass


def calculate_means(
        dataset: List[Rollout],
        active_tasks: List[Task]
):
    cumulative_rewards = [0.0] * len(active_tasks)
    counts = [0] * len(active_tasks)
    for rollout in dataset:
        cumulative_rewards[rollout.task_idx] += sum(sample.reward for sample in rollout.trajectory)
        counts[rollout.task_idx] += 1
    means = torch.tensor([float(cumulative_rewards[i])/counts[i] for i in range(len(active_tasks))])
    return means


def train_step(
        model: GridWorldModel,
        active_tasks: List[Task],
        curriculum,
        max_num_rollouts=100,
):
    # Generate dataset of rollouts given current curriculum over tasks.
    dataset: List[Rollout] = []
    while len(dataset) < max_num_rollouts:
        task: Task = active_tasks[curriculum.sample().item()]
        rollout: Rollout = do_rollout(task, model)
        dataset.append(rollout)
    for rollout in dataset:
        model.update_policy()
        model.update_critic()

    return calculate_means(dataset, tasks)


"""
TODO:
train_loop

train_step
- sample task
- do rollouts
- update policy / critic
"""

def update_curriculum(
        means: torch.tensor
):
    return dist.Categorical(1 - means)

def train_loop(
        model: GridWorldModel,
        tasks: List[Task],
        max_task_complexity=50,
        good_task_reward=0.7,
):
    curr_task_complexity = 1
    while curr_task_complexity < max_task_complexity:
        min_task_reward = float("-inf")
        active_tasks: List[Task] = [
            task for task in tasks if task.complexity < curr_task_complexity
        ]
        # Curriculum is distribution `torch.Categorical[N_active_tasks]``
        curriculum = dist.Categorical(torch.ones(len(active_tasks)))
        while min_task_reward < good_task_reward:
            means = train_step(model, active_tasks, curriculum)
            curriculum = update_curriculum(means)
            min_task_reward = torch.min(means)
        curr_task_complexity += 1


# From Andreas et al.
NUM_ROWS = 10
NUM_COLS = 10

Task = namedtuple("Task", ["goal", "world", "complexity"])


def main():
    tasks: List[Task] = [
        Task("MakePlank",  GridWorld(NUM_ROWS, NUM_COLS, Item.PLANK), 2),
        Task("MakeStick",  GridWorld(NUM_ROWS, NUM_COLS, Item.STICK), 2),
        Task("MakeCloth",  GridWorld(NUM_ROWS, NUM_COLS, Item.CLOTH), 2),
        Task("MakeRope",   GridWorld(NUM_ROWS, NUM_COLS, Item.ROPE), 2),
        Task("MakeBridge", GridWorld(NUM_ROWS, NUM_COLS, Item.BRIDGE), 3),
        Task("MakeShears", GridWorld(NUM_ROWS, NUM_COLS, Item.SHEARS), 3),
    ]
    model = GridWorldModel(pg, tasks)
    train_loop(model, tasks)


if __name__ == "__main__":
    main()
