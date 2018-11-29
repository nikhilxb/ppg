import argparse
from collections import namedtuple
from typing import List, Any

import pendulum
import torch
import torch.distributions as dist
import torch.nn as nn

from models import GridWorldAgent
from ppg.grammar import PolicyGrammar
from worlds.gridworld import Item, GridWorld


# ==================================================================================================
# Definitions.


def define_args() -> Any:
    parser = argparse.ArgumentParser()

    # Experiment options.
    parser.add_argument(
        "--experiment_name",
        default="exp-{}".format(pendulum.now("America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S")),
        help="Name of directory to save experiment outputs.",
    )
    parser.add_argument(
        "--experiments_dir",
        help="Directory where all experiment directories ar stored.",
        default="experiments/",
    )

    # GridWorld options.
    parser.add_argument("--grid_row_num", type=int, default=10)
    parser.add_argument("--grid_col_num", type=int, default=10)

    # Agent options.
    parser.add_argument("--agent_state_dim", type=int, default=32)
    parser.add_argument("--agent_action_dim", type=int, default=32)
    parser.add_argument("--activation_net_hidden_dim", type=int, default=32)
    parser.add_argument("--production_net_hidden_dim", type=int, default=32)
    parser.add_argument("--policy_net_hidden_dim", type=int, default=32)
    parser.add_argument("--state_net_hidden_dim", type=int, default=32)
    parser.add_argument("--critic_net_hidden_dim", type=int, default=32)

    return parser.parse_args()


def define_grammar() -> PolicyGrammar:
    grammar = PolicyGrammar(
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

    # yapf: disable
    pi, g = grammar.get_tokens()
    grammar.add_productions("MakePlank",  [pi["GetWood"], pi["UseToolshed"]])
    grammar.add_productions("MakeStick",  [pi["GetWood"], pi["UseWorkbench"]])
    grammar.add_productions("MakeCloth",  [pi["GetGrass"], pi["UseFactory"]])
    grammar.add_productions("MakeRope",   [pi["GetGrass"], pi["UseToolshed"]])
    grammar.add_productions("MakeBridge", [pi["GetIron"], pi["GetWood"], pi["UseFactory"]])
    grammar.add_productions("MakeShears", [g["MakeStick"], pi["UseToolshed"]])
    grammar.add_productions("MakeAxe",    [g["MakeStick"], pi["GetIron"], pi["UseToolshed"]])
    grammar.add_productions("MakeBed",    [g["MakePlank"], pi["GetGrass"], pi["UseWorkbench"]])
    grammar.add_productions("MakeLadder", [g["MakePlank"], g["MakeStick"], pi["UseFactory"]])
    # yapf: enable
    return grammar


# A task attempts to solve a particular world using a policy constructed for a particular
# top-level goal. The complexity of a task is how many primitives need to be sequentially executed.
Task = namedtuple("Task", ["goal", "world", "complexity"])


def define_tasks(args) -> List[Task]:
    # yapf: disable
    tasks = [
        Task("MakePlank",  GridWorld(args.grid_row_num, args.grid_col_num, Item.PLANK),  2),
        Task("MakeStick",  GridWorld(args.grid_row_num, args.grid_col_num, Item.STICK),  2),
        Task("MakeCloth",  GridWorld(args.grid_row_num, args.grid_col_num, Item.CLOTH),  2),
        Task("MakeRope",   GridWorld(args.grid_row_num, args.grid_col_num, Item.ROPE),   2),
        Task("MakeBridge", GridWorld(args.grid_row_num, args.grid_col_num, Item.BRIDGE), 3),
        Task("MakeShears", GridWorld(args.grid_row_num, args.grid_col_num, Item.SHEARS), 3),
    ]
    # yapf: enable
    return tasks


# ==================================================================================================
# Training process.

# Experience gained from a single transition.
#     state:  torch.Tensor
#     action: gridworld.Action
#     reward: float
Transition = namedtuple("Transition", ["state", "action", "reward"])

# An entire rollout of transitions for an episode.
Trajectory = List[Transition]

Rollout = namedtuple("Rollout", ["task_idx", "trajectory"])

# task_idx: int
# trajectory: List[Sample]


def do_rollout(model: GridWorldAgent, task: Task):
    # Use the algorithm to rollout a trajectory, given the task
    pass


def calculate_means(dataset: List[Rollout], active_tasks: List[Task]):
    cumulative_rewards = [0.0] * len(active_tasks)
    counts = [0] * len(active_tasks)
    for rollout in dataset:
        cumulative_rewards[rollout.task_idx] += sum(sample.reward for sample in rollout.trajectory)
        counts[rollout.task_idx] += 1
    means = torch.tensor(
        [float(cumulative_rewards[i]) / counts[i] for i in range(len(active_tasks))]
    )
    return means


def train_step(
        model: GridWorldAgent,
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


def update_curriculum(means: torch.tensor):
    return dist.Categorical(1 - means)


def train_loop(
        model: GridWorldAgent,
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


# ==================================================================================================


def main():
    args = define_args()
    grammar: PolicyGrammar = define_grammar()
    tasks: List[Task] = define_tasks(args)
    agent: GridWorldAgent = GridWorldAgent(
        grammar,
        args.agent_state_dim,
        args.agent_action_dim,
        args.activation_net_hidden_dim,
        args.production_net_hidden_dim,
        args.policy_net_hidden_dim,
        args.state_net_hidden_dim,
    )
    critic: nn.Module = nn.Sequential(
        nn.Linear(args.agent_state_dim, args.critic_net_hidden_dim),
        nn.ReLU(),
        nn.Linear(args.policy_net_hidden_dim, agent_action_dim),
    )

    train_loop(agent, tasks)


if __name__ == "__main__":
    main()
