import argparse
from collections import namedtuple
from typing import List, Mapping, Any

import random
import pendulum
import torch
import torch.distributions as dist
import torch.nn as nn

from models import GridWorldAgent
from ppg.grammar import PolicyGrammar
from worlds.gridworld import GridWorld, Item, Observation, Action


# ==================================================================================================
# Definitions.


# Args mapping accessible from all methods in file.
args = None


def define_args() -> Any:
    parser = argparse.ArgumentParser()

    # Experiment options.
    parser.add_argument(
        "--experiment_name",
        default="exp-{}".format(make_timestamp()),
        help="Name of directory to save experiment outputs.",
    )
    parser.add_argument(
        "--experiments_dir",
        help="Directory where all experiment directories ar stored.",
        default="experiments/",
    )
    parser.add_argument("--seed", type=int, default=0)

    # GridWorld options.
    parser.add_argument("--world_num_rows", type=int, default=10)
    parser.add_argument("--world_num_cols", type=int, default=10)
    parser.add_argument("--world_max_timesteps", type=int, default=30)

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
Task = namedtuple("Task", ["grammar_goal", "target_item", "complexity"])


def define_tasks(args) -> Mapping[int, Task]:
    # yapf: disable
    tasks = [
        Task("MakePlank",  Item.PLANK,  2),
        Task("MakeStick",  Item.STICK,  2),
        Task("MakeCloth",  Item.CLOTH,  2),
        Task("MakeRope",   Item.ROPE,   2),
        Task("MakeBridge", Item.BRIDGE, 3),
        Task("MakeShears", Item.SHEARS, 3),
    ]
    # yapf: enable
    return {i: task for i, task in enumerate(tasks)}


# ==================================================================================================
# Training process.


# Experience gained from a single transition.
#     state:  torch.Tensor
#     action: gridworld.Action
#     reward: float
Transition = namedtuple("Transition", ["state", "action", "reward"])


# An episode rollout of transitions for a given task.
#     task_idx: int
#     trajectory: List[Transition]
Rollout = namedtuple("Rollout", ["task_idx", "trajectory"])


def observation_to_tensor(observation: Observation) -> torch.Tensor:
    raise NotImplementedError


def action_probs_to_action(action_probs: torch.Tensor) -> Action:
    raise NotImplementedError


def do_rollout(world: GridWorld, agent: GridWorldAgent, goal: str) -> List[Transition]:
    rollout: List[Transition] = []
    agent.reset()
    observation: Observation = world.reset()
    while True:
        agent_state, action_probs = agent.forward(observation_to_tensor(observation), goal)
        action: Action = action_probs_to_action(action_probs)
        observation, reward, done, info = world.step(action)
        rollout.append(Transition(state=agent_state, action=action, reward=reward))
        if done: break
    return rollout


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
        agent: GridWorldAgent,
        active_tasks: List[Task],
        curriculum,
        max_num_rollouts=100,
):
    # Generate dataset of rollouts given current curriculum over tasks.
    dataset: List[List[Transition]] = []
    while len(dataset) < max_num_rollouts:
        task: Task = active_tasks[curriculum.sample().item()]
        world: GridWorld = GridWorld(args.world_num_rows, args.world_num_cols, task.target_item)
        rollout: List[Transition] = do_rollout(world, agent, task.grammar_goal)
        dataset.append(rollout)

    # Update model parameters given dataset.
    for rollout in dataset:
        # Actor.
        pass

        # Critic.
        pass

    return calculate_means(dataset, tasks)


def train_loop(
        agent: GridWorldAgent,
        critic: torch.Module,
        tasks: List[Task],
        max_task_complexity=3,
        good_task_reward=0.7,
):
    curr_task_complexity = 1
    while curr_task_complexity <= max_task_complexity:
        min_task_reward = float("-inf")
        active_tasks: List[Task] = [t for t in tasks if t.complexity < curr_task_complexity]

        # Curriculum is distribution `torch.Categorical[N_active_tasks]``
        curriculum = dist.Categorical(torch.ones(len(active_tasks)))
        while min_task_reward < good_task_reward:
            means = train_step(agent, active_tasks, curriculum)
            curriculum = dist.Categorical(1 - means)
            min_task_reward = torch.min(means)
        curr_task_complexity += 1


# ==================================================================================================
# Utility functions.


def make_timestamp() -> str:
    return pendulum.now("America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S")


def save_model(name: str, model, timestamp=True):
    raise NotImplementedError  # TODO


def configure():
    random.seed(args.seed)
    torch.manual_seed(args.seed)


def main():
    global args
    args = define_args()
    configure()

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
        nn.Linear(args.policy_net_hidden_dim, len(tasks)),
    )

    train_loop(agent, critic, tasks)
    save_model("agent-final", agent)
    save_model("critic-final", critic)


if __name__ == "__main__":
    main()
