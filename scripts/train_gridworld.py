import argparse
import random
import pendulum
import torch
import torch.cuda
import torch.distributions as dist
import torch.nn as nn

from collections import namedtuple, defaultdict
from typing import List, Tuple, Mapping, Any

from ppg.grammar import PolicyGrammar
from ppg.models import GridWorldAgent
from worlds.gridworld import GridWorld, Item, Observation, Action

# ==================================================================================================
# Definitions.

# Args mapping accessible from all methods in file.
args = None


def define_args() -> Any:
    parser = argparse.ArgumentParser()

    # Experiment options.
    parser.add_argument("--experiment_name", default="exp-{}".format(make_timestamp()))
    parser.add_argument("--experiments_dir", default="experiments/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=50)

    # GridWorld options.
    parser.add_argument("--world_num_rows", type=int, default=10)
    parser.add_argument("--world_num_cols", type=int, default=10)
    parser.add_argument("--world_max_timesteps", type=int, default=30)
    parser.add_argument("--world_window_radius", type=int, default=2)

    # Agent options.
    parser.add_argument("--agent_state_dim", type=int, default=128)
    parser.add_argument("--agent_action_dim", type=int, default=5)
    parser.add_argument("--activation_net_hidden_dim", type=int, default=32)
    parser.add_argument("--production_net_hidden_dim", type=int, default=32)
    parser.add_argument("--policy_net_hidden_dim", type=int, default=64)
    parser.add_argument("--state_net_hidden_dim", type=int, default=128)
    parser.add_argument("--critic_net_hidden_dim", type=int, default=64)

    # Training options.
    parser.add_argument("--num_rollouts", type=int, default=2000)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--task_reward_threshold", type=float, default=0.8)

    return parser.parse_args()


def define_grammar() -> PolicyGrammar:
    grammar: PolicyGrammar = PolicyGrammar(
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


def define_tasks(args) -> Tuple[Task]:
    # yapf: disable
    tasks: Tuple[Task] = (
        Task("MakePlank",  Item.PLANK,  2),
        Task("MakeStick",  Item.STICK,  2),
        Task("MakeCloth",  Item.CLOTH,  2),
        Task("MakeRope",   Item.ROPE,   2),
        Task("MakeBridge", Item.BRIDGE, 3),
        Task("MakeShears", Item.SHEARS, 3),
    )
    # yapf: enable
    return tasks


# ==================================================================================================
# Training process.

# Experience gained from a single transition.
#     state:  torch.Tensor
#     action: gridworld.Action
#     reward: float
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "log_prob", "critic_value", "discounted_return"]
)

# An episode rollout of transitions for a given task.
#     task_idx: int
#     trajectory: List[Transition]
Rollout = namedtuple("Rollout", ["task_idx", "trajectory"])


def encode_observation(observation: Observation) -> torch.Tensor:
    raise NotImplementedError  # TODO


def generate_trajectory(
        world: GridWorld,
        agent: GridWorldAgent,
        goal: str,
        critic: torch.Module,
        task_idx: int,
        deterministic: bool = False,
) -> List[Transition]:
    trajectory: List[Transition] = []

    # Perform typical RL loop.
    agent.reset()
    obs: Observation = world.reset()
    while True:
        agent_state, action_probs = agent(encode_observation(obs), goal)
        action_dist = dist.Categorical(action_probs)
        action_idx = action_dist.sample() if not deterministic else action_probs.argmax()

        log_prob = action_dist.log_prob(action_idx)
        critic_value = critic(agent_state)[task_idx]

        action: Action = Action(action_idx.item())
        obs, reward, done, info = world.step(action)

        trajectory.append(
            Transition(
                state=agent_state,
                action=action,
                reward=reward,
                log_prob=log_prob,
                critic_value=critic_value,
                discounted_return=None,
            )
        )
        if done: break

    # Compute discounted returns from reward of each transition.
    curr_return = 0.0
    for t in reversed(range(len(trajectory))):
        curr_return = trajectory[t].reward + args.discount_factor * curr_return
        trajectory[t] = trajectory[t]._replace(discounted_return=curr_return)

    return trajectory


def train_step(
        agent: GridWorldAgent,
        critic: torch.Module,
        active_tasks: Tuple[Tuple[int, Task]],
        curriculum: torch.Categorical,
        num_rollouts: int = 100,
):
    # Generate dataset of rollouts given current curriculum over tasks.
    dataset: List[Rollout] = []
    while len(dataset) < num_rollouts:
        task_idx, task = active_tasks[curriculum.sample().item()]
        world: GridWorld = GridWorld(
            args.world_num_rows,
            args.world_num_cols,
            task.target_item,
            max_timesteps=args.world_max_timesteps,
            window_radius=args.world_window_radius,
        )
        trajectory: List[Transition] = generate_trajectory(
            world, agent, task.grammar_goal, critic, task_idx
        )
        if len(trajectory) == 0: continue
        dataset.append(Rollout(task_idx=task_idx, trajectory=trajectory))

    # Update model parameters given dataset.
    task_reward_sum = defaultdict(float)
    task_count = defaultdict(int)
    for rollout in dataset:
        state_arr, action_arr, reward_arr, log_prob_arr, critic_value_arr, discounted_return_arr \
            = zip(*rollout.trajectory)

        # TODO: PPO algorithm
        raise NotImplementedError

        task_reward_sum[rollout.task_idx] += sum(reward_arr)
        task_count[rollout.task_idx] += 1
    means = torch.tensor(
        [float(cumulative_rewards[i]) / counts[i] for i in range(len(active_tasks))]
    )
    return {task_idx: torch for task_idx in task_c}


def train_loop(
        agent: GridWorldAgent,
        critic: torch.Module,
        tasks: Tuple[Task],
        max_task_complexity: int = 3,
) -> None:
    curr_task_complexity = 1
    while curr_task_complexity <= max_task_complexity:
        min_task_reward = float("-inf")
        active_tasks: Tuple[Tuple[int, Task]] = tuple(
            [
                (task_idx, task)
                for task_idx, task in enumerate(tasks)
                if task.complexity < curr_task_complexity
            ]
        )
        if len(active_tasks) == 0:
            curr_task_complexity += 1
            continue

        # Curriculum is multinomial `torch.Categorical[len(active_tasks)]`.
        curriculum = dist.Categorical(torch.ones(len(active_tasks)))
        while min_task_reward < args.task_reward_threshold:
            means = train_step(
                agent, critic, active_tasks, curriculum, num_rollouts=args.num_rollouts
            )
            curriculum = dist.Categorical(1 - means)
            min_task_reward = torch.min(means)
        curr_task_complexity += 1


# ==================================================================================================
# Utility functions.


def make_timestamp() -> str:
    return pendulum.now("America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S")


def save_model(name: str, model, timestamp=True):
    raise NotImplementedError  # TODO


def configure() -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main() -> None:
    global args
    args = define_args()
    configure()

    grammar: PolicyGrammar = define_grammar()
    tasks: Tuple[Task] = define_tasks(args)
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
