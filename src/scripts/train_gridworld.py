import argparse
import random
import pendulum
import logging
import os
import sys
import traceback
import tqdm
import torch
import torch.cuda
import torch.distributions as dist
import torch.nn as nn

from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from dataclasses import dataclass
from textwrap import indent
from typing import List, Tuple, Sequence, Mapping, Any, cast

from ppg.grammar import PolicyGrammar
from ppg.models import PolicyGrammarAgent, PolicySketchAgent
from worlds.gridworld import GridWorld, Item, Observation, Action, Cell
from utils.rl import (Sample, Trajectory, PPOClipLoss, compute_returns, compute_advantages)
from utils.data import SequenceDataset

# ==================================================================================================
# Definitions.

# Args mapping accessible from all methods in file.
args = None


def define_args() -> None:
    parser = argparse.ArgumentParser()

    # Experiment options.
    parser.add_argument("experiment_name")
    parser.add_argument("--experiments_dir", default="experiments/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--checkpoint_interval", type=int, default=50)

    # GridWorld options.
    parser.add_argument("--world_num_rows", type=int, default=10)
    parser.add_argument("--world_num_cols", type=int, default=10)
    parser.add_argument("--world_max_timesteps", type=int, default=100)
    parser.add_argument("--world_window_radius", type=int, default=2)

    # Agent options.
    parser.add_argument("--agent_type", type=str, default="ppg", choices=["ppg", "sketch"])
    parser.add_argument("--agent_state_dim", type=int, default=128)
    parser.add_argument("--agent_action_dim", type=int, default=5)
    parser.add_argument("--activation_net_hidden_dim", type=int, default=32)
    parser.add_argument("--production_net_hidden_dim", type=int, default=32)
    parser.add_argument("--policy_net_hidden_dim", type=int, default=64)
    parser.add_argument("--state_net_layers_num", type=int, default=1)
    parser.add_argument("--critic_net_hidden_dim", type=int, default=64)

    # Dataset generation options.
    parser.add_argument("--num_rollouts", type=int, default=1024)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--task_reward_threshold", type=float, default=0.8)

    # PPO options.
    parser.add_argument("--ppo_num_epochs", type=int, default=3)
    parser.add_argument("--ppo_minibatch_size", type=int, default=128)
    parser.add_argument("--ppo_clip_epsilon", type=float, default=0.2)
    parser.add_argument("--ppo_value_loss_coeff", type=float, default=0.9)
    parser.add_argument("--ppo_entropy_loss_coeff", type=float, default=0.1)
    parser.add_argument("--adam_lr", type=float, default=3e-4)

    global args
    args = parser.parse_args()


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
            # "MakeAxe",
            # "MakeBed",
            # "MakeLadder",`
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
    # grammar.add_productions("MakeAxe",    [g["MakeStick"], pi["GetIron"], pi["UseToolshed"]])
    # grammar.add_productions("MakeBed",    [g["MakePlank"], pi["GetGrass"], pi["UseWorkbench"]])
    # grammar.add_productions("MakeLadder", [g["MakePlank"], g["MakeStick"], pi["UseFactory"]])
    # yapf: enable
    return grammar


def define_sketches() -> Mapping[str, Sequence[str]]:
    # yapf: disable
    return {
        "MakePlank":  ["GetWood", "UseToolshed"],
        "MakeStick":  ["GetWood", "UseWorkbench"],
        "MakeCloth":  ["GetGrass", "UseFactory"],
        "MakeRope":   ["GetGrass", "UseToolshed"],
        "MakeBridge": ["GetIron", "GetWood", "UseFactory"],
        "MakeShears": ["GetWood", "UseWorkbench", "UseToolshed"],
        # "MakeAxe":    ["GetWood", "UseWorkbench", "GetIron", "UseToolshed"],
        # "MakeBed":    ["GetWood", "UseToolshed", "GetGrass", "UseWorkbench"],
        # "MakeLadder": ["GetWood", "UseToolshed", "GetWood", "UseWorkbench", "UseFactory"],
    }
    # yapf: enable


@dataclass
class Task:
    """A task attempts to solve a particular world using a policy constructed for a particular
    top-level goal. Task complexity is how many primitives need to be sequentially executed.
    """
    grammar_goal: str
    target_item: Item
    complexity: int


def define_tasks() -> Sequence[Task]:
    # yapf: disable
    tasks: Sequence[Task] = (
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


@dataclass
class Trial:
    """An single episode rollout on a given task."""
    active_task_idx: int
    task_idx: int
    task: Task
    rollout: Trajectory


def calc_observation_dim(window_radius: int) -> int:
    return len(Item) + len(Cell) * (2 * window_radius + 1)**2


def encode_observation(obs_raw: Observation) -> torch.Tensor:
    inventory, window = obs_raw

    inventory_tensor = torch.zeros(len(Item))
    for item, count in inventory.items():
        inventory_tensor[item.value] = min(count, 1)

    num_rows, num_cols = len(window), len(window[0])
    window_tensor = torch.zeros(num_rows, num_cols, len(Cell))
    for r in range(num_rows):
        for c in range(num_cols):
            if window[r][c] is not None:
                window_tensor[r, c, window[r][c].value] = 1

    return torch.cat((inventory_tensor, window_tensor.flatten()))


def generate_rollout(
        world: GridWorld,
        agent: nn.Module,
        grammar_goal: str,
        critic: nn.Module,
        task_idx: int,
        deterministic: bool = False,
) -> Trajectory:
    samples: Sequence[Sample] = []

    # Perform typical RL loop.
    agent.reset(grammar_goal, device=args.device)
    obs_raw: Observation = world.reset()
    while True:
        obs = encode_observation(obs_raw).to(args.device)  # size[D]

        primitive_idx = None
        if args.agent_type == "ppg":
            agent_state, action_probs = agent(obs.unsqueeze(0))  # size[1, *]
        elif args.agent_type == "sketch":
            agent_state, action_probs, primitive_idx = agent(obs.unsqueeze(0))

        state_value = critic(agent_state)[:, task_idx]

        action_probs = action_probs.squeeze(0)
        state_value = state_value.squeeze(0)

        action_dist = dist.Categorical(action_probs)
        action = action_dist.sample() if not deterministic else action_probs.argmax()
        log_prob = action_dist.log_prob(action)

        action_raw: Action = Action(action.item())
        obs_raw, reward_raw, done, info = world.step(action_raw)
        reward = torch.tensor(float(reward_raw)).to(args.device)

        # Must detach results computed by neural networks from computational graph as these values
        # are just used to compute gradients for the model. We don't actually want the gradients to
        # be propagating through them.
        samples.append(
            Sample(
                obs=obs,
                action=action,
                reward=reward,
                log_prob=log_prob.detach(),
                state_value=state_value.detach(),
                ret=None,
                advantage=None,
                primitive_idx=primitive_idx,
            )
        )
        if done: break

    samples = compute_returns(samples, discount=args.discount_factor, device=args.device)
    samples = compute_advantages(samples)

    return Trajectory(samples)


def do_ppo_updates(
        agent: PolicyGrammarAgent,
        critic: nn.Module,
        dataset: Sequence[Trial],
        num_epochs: int = 3,
        minibatch_size: int = 32,
):
    optimizer: Optimizer = Adam(
        list(agent.parameters()) + list(critic.parameters()), lr=args.adam_lr
    )
    criterion = PPOClipLoss(
        clip_epsilon=args.ppo_clip_epsilon,
        value_loss_coeff=args.ppo_value_loss_coeff,
        entropy_loss_coeff=args.ppo_entropy_loss_coeff,
    )

    # Perform multiple passes over the dataset, increasing sample efficiency.
    for epoch in range(num_epochs):
        logging.info("Performing PPO updates; epoch {}.".format(epoch))
        dataloader: DataLoader = DataLoader(
            SequenceDataset(dataset),
            batch_size=minibatch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=lambda batch: batch,
        )

        # Update model parameters after each minibatch of gradient calculations.
        for minibatch in tqdm.tqdm(dataloader):
            minibatch_loss = 0.0
            for trial in minibatch:
                trial = cast(Trial, trial)
                grammar_goal: str = trial.task.grammar_goal
                task_idx: int = trial.task_idx
                rollout: Trajectory = trial.rollout

                agent.reset(grammar_goal, device=args.device)

                log_prob_old = rollout.log_prob  # size[t]
                advantage_old = rollout.advantage
                ret = rollout.ret

                if args.agent_type == "ppg":
                    agent_state, action_probs = agent(rollout.obs)
                elif args.agent_type == "sketch":
                    agent_state, action_probs = agent.evaluate(rollout.obs, rollout.primitive_idx)

                state_value = critic(agent_state)[:, task_idx]

                action_dist = dist.Categorical(action_probs)
                log_prob = action_dist.log_prob(rollout.action)
                entropy = action_dist.entropy()

                minibatch_loss += criterion(
                    log_prob,
                    log_prob_old,
                    advantage_old,
                    state_value,
                    ret,
                    entropy,
                )
            optimizer.zero_grad()
            minibatch_loss.backward()
            optimizer.step()


def train_step(
        agent: nn.Module,
        critic: nn.Module,
        active_tasks: Sequence[Tuple[int, Task]],
        curriculum: dist.Categorical,
        num_rollouts: int = 1000,
):
    # Generate dataset of rollouts given current curriculum over tasks.
    dataset: List[Trial] = []
    logging.info("Generating dataset of {} rollouts.".format(num_rollouts))
    for n in tqdm.tqdm(range(num_rollouts)):
        active_task_idx: int = curriculum.sample().item()
        task_idx, task = active_tasks[active_task_idx]
        world: GridWorld = GridWorld(
            args.world_num_rows,
            args.world_num_cols,
            task.target_item,
            max_timesteps=args.world_max_timesteps,
            window_radius=args.world_window_radius,
        )
        rollout: Trajectory = generate_rollout(world, agent, task.grammar_goal, critic, task_idx)
        dataset.append(
            Trial(
                active_task_idx=active_task_idx,
                task_idx=task_idx,
                task=task,
                rollout=rollout,
            )
        )

    # Estimate average reward per task to use for updating curriculum.
    reward_sums: torch.Tensor = torch.zeros(len(active_tasks))
    counts: torch.Tensor = torch.ones(len(active_tasks))  # Prevent divide by 0.
    for trial in dataset:
        reward_sums[trial.active_task_idx] += trial.rollout.reward.sum()
        counts[trial.active_task_idx] += 1

    # Record average reward over all tasks.
    reward_avg_all = reward_sums.sum() / counts.sum()
    logging.info(
        "Generated dataset of {} rollouts; avg reward = {}. ".format(
            len(dataset), reward_avg_all.item()
        )
    )

    # Update model parameters by performing multiple epochs of PPO.
    do_ppo_updates(
        agent,
        critic,
        dataset,
        num_epochs=args.ppo_num_epochs,
        minibatch_size=args.ppo_minibatch_size,
    )

    return reward_sums, counts


def train_loop(
        agent: nn.Module,
        critic: nn.Module,
        tasks: Sequence[Task],
        max_task_complexity: int = 3,
) -> None:
    num_steps: int = 0
    curr_task_complexity: int = 1
    while curr_task_complexity <= max_task_complexity:
        active_tasks: Sequence[Tuple[int, Task]] = tuple(
            [
                (task_idx, task)
                for task_idx, task in enumerate(tasks)
                if task.complexity <= curr_task_complexity
            ]
        )

        logging.info(
            "Training loop started for tasks with complexity <= {}; active tasks = {}".format(
                curr_task_complexity, len(active_tasks)
            )
        )

        if len(active_tasks) == 0:
            curr_task_complexity += 1
            continue

        # Curriculum is multinomial `dist.Categorical[len(active_tasks)]`.
        curriculum = dist.Categorical(torch.ones(len(active_tasks)))
        min_task_reward: float = float("-inf")
        while min_task_reward < args.task_reward_threshold:
            reward_sums, counts = train_step(
                agent, critic, active_tasks, curriculum, num_rollouts=args.num_rollouts
            )
            avg_task_reward: float = (reward_sums.sum() / counts.sum()).item()
            means = reward_sums / counts
            curriculum = dist.Categorical(1 - means)
            min_task_reward = torch.min(means).item()
            num_steps += 1
            logging.info(
                "Training step {} complete; min task reward = {}".format(
                    num_steps, min_task_reward
                )
            )

            # Checkpoint model parameters once in a while.
            if num_steps % args.checkpoint_interval == 0:
                save_state(
                    "checkpoint", {
                        "agent_state_dict": agent.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "min_task_reward": min_task_reward,
                        "num_steps": num_steps,
                        "avg_task_reward": avg_task_reward,
                    }
                )

            # Save results on every step.
            args.results.write("{} {} {}\n".format(num_steps, avg_task_reward, min_task_reward))
            args.results.flush()

        curr_task_complexity += 1


# ==================================================================================================
# Utility functions.


def make_timestamp() -> str:
    return pendulum.now("America/Los_Angeles").strftime("%Y-%m-%d-%H-%M-%S")


def save_state(filename: str, state_dict: Mapping[str, Any], timestamp=True):
    torch.save(
        state_dict,
        os.path.join(
            args.working_dir,
            "{}{}.pkl".format(filename, "-{}".format(make_timestamp()) if timestamp else "")
        )
    )


def configure() -> None:
    # Set random seeds.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set CPU or GPU.
    args.device = "cuda:0" if args.cuda else "cpu"
    if args.cuda: torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Set up experiment directory.
    if not os.path.exists(args.experiments_dir): os.mkdir(args.experiments_dir)
    args.working_dir = os.path.join(
        args.experiments_dir, "{}-{}".format(args.experiment_name, make_timestamp())
    )
    assert not os.path.exists(args.working_dir)
    os.mkdir(args.working_dir)

    # Set up results output.
    args.results = open(os.path.join(args.working_dir, "results.txt"), "w")

    # Set up logging.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.working_dir, "logging.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = lambda t, v, tb: logging.exception(
        "".join(traceback.format_exception(t, v, tb))
    )

    # Log configuration.
    logging.info(
        "Starting experiment with configuration:\n{}".format(
            indent("\n".join("{} = {}".format(k, v) for k, v in args.__dict__.items()), " " * 4)
        )
    )


def main() -> None:
    define_args()
    configure()

    tasks: Sequence[Task] = define_tasks()

    if args.agent_type == "ppg":
        grammar: PolicyGrammar = define_grammar()
        agent: PolicyGrammarAgent = PolicyGrammarAgent(
            grammar,
            env_observation_dim=calc_observation_dim(args.world_window_radius),
            agent_state_dim=args.agent_state_dim,
            agent_action_dim=args.agent_action_dim,
            activation_net_hidden_dim=args.activation_net_hidden_dim,
            production_net_hidden_dim=args.production_net_hidden_dim,
            policy_net_hidden_dim=args.policy_net_hidden_dim,
            state_net_layers_num=args.state_net_layers_num,
        ).to(args.device)
    elif args.agent_type == "sketch":
        sketch: Mapping[str, Sequence[str]] = define_sketches()
        agent: PolicySketchAgent = PolicySketchAgent(
            sketch,
            env_observation_dim=calc_observation_dim(args.world_window_radius),
            agent_state_dim=args.agent_state_dim,
            agent_action_dim=args.agent_action_dim,
            policy_net_hidden_dim=args.policy_net_hidden_dim,
            state_net_layers_num=args.state_net_layers_num,
        ).to(args.device)

    critic: nn.Module = nn.Sequential(
        nn.Linear(args.agent_state_dim, args.critic_net_hidden_dim),
        nn.ReLU(),
        nn.Linear(args.policy_net_hidden_dim, len(tasks)),
    ).to(args.device)

    try:
        train_loop(agent, critic, tasks)
    except KeyboardInterrupt:
        logging.info("Training interrupted by keyboard command.")

    save_state(
        "final", {
            "agent_state_dict": agent.state_dict(),
            "critic_state_dict": critic.state_dict(),
        }
    )
    args.results.flush()
    args.results.close()


if __name__ == "__main__":
    main()
