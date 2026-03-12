"""
Training Script for Q-Learning Agent

Single entry point to configure, train, evaluate, and save the Q-Learning agent.
Run: python training/train_qlearning.py  OR  python -m training.train_qlearning
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze_env import MazeEnv
from agents.q_learning import QLearningAgent, train, evaluate, plot_metrics
from utils.visualization import visualize_single, record_agent_path, visualize_training_progress


# ─── Configuration ───────────────────────────────────────────────────────────

ENV_CONFIG = {
    "width": 21,
    "height": 21,
    "trap_fraction": 0.10,
    "wall_removal_fraction": 0.3,
    "max_steps": 500,
    "seed": 42,
}

AGENT_CONFIG = {
    "action_size": 4,
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
}

TRAIN_CONFIG = {
    "episodes": 1000,
    "eval_episodes": 100,
    "snapshot_episodes": [1, 50, 100, 200, 500, 1000],
    "multi_maze_seeds": [42, 99, 7, 123, 256],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "q_learning")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_config(path):
    """Save all configs to a JSON file for reproducibility."""
    config = {
        "env": ENV_CONFIG,
        "agent": AGENT_CONFIG,
        "training": TRAIN_CONFIG,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def collect_snapshots(env, agent, total_episodes, snapshot_episodes):
    """
    Train in chunks, recording the agent's greedy path at snapshot episodes.

    Args:
        env: MazeEnv instance.
        agent: QLearningAgent instance.
        total_episodes: Total training episodes.
        snapshot_episodes: Sorted list of episode numbers to snapshot (1-indexed).

    Returns:
        Tuple of (all_metrics, snapshots list).
    """
    all_metrics = {"rewards": [], "steps": [], "successes": [], "epsilons": []}
    snapshots = []
    trained_so_far = 0

    for snap_ep in snapshot_episodes:
        chunk = snap_ep - trained_so_far
        if chunk > 0:
            metrics = train(env, agent, episodes=chunk)
            for key in all_metrics:
                all_metrics[key].extend(metrics[key])
            trained_so_far = snap_ep

        path = record_agent_path(env, agent)
        reached_exit = (path[-1] == env.exit_pos)
        snapshots.append({
            "episode": snap_ep,
            "path": path,
            "steps": len(path) - 1,
            "solved": reached_exit,
        })
        print(f"  Snapshot @ episode {snap_ep}: {len(path)-1} steps, "
              f"{'SOLVED' if reached_exit else 'FAILED'}")

    remaining = total_episodes - trained_so_far
    if remaining > 0:
        metrics = train(env, agent, episodes=remaining)
        for key in all_metrics:
            all_metrics[key].extend(metrics[key])

    return all_metrics, snapshots


def train_single_maze():
    """Train and evaluate on a single maze with training progress snapshots."""
    print("=" * 60)
    print("  Q-Learning — Single Maze Training")
    print("=" * 60)

    env = MazeEnv(**ENV_CONFIG)
    agent = QLearningAgent(**AGENT_CONFIG)

    train_metrics, snapshots = collect_snapshots(
        env, agent,
        total_episodes=TRAIN_CONFIG["episodes"],
        snapshot_episodes=TRAIN_CONFIG["snapshot_episodes"]
    )

    print("\n" + "-" * 60)
    eval_metrics = evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "q_table_single.pkl"))
    plot_metrics(train_metrics, save_path=os.path.join(OUTPUT_DIR, "single_maze_metrics.png"))

    print("\n" + "-" * 60)
    print("Visualizing trained agent...")
    visualize_single(env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "q_learning_single.gif"))

    print("Visualizing training progress...")
    visualize_training_progress(env, snapshots,
                                save_path=os.path.join(OUTPUT_DIR, "q_learning_progress.gif"))

    return agent, train_metrics, eval_metrics


def train_multi_maze():
    """Train across multiple procedurally generated mazes for generalization."""
    print("\n" + "=" * 60)
    print("  Q-Learning — Multi-Maze Training")
    print("=" * 60)

    seeds = TRAIN_CONFIG["multi_maze_seeds"]
    episodes_per_maze = TRAIN_CONFIG["episodes"] // len(seeds)
    agent = QLearningAgent(**AGENT_CONFIG)

    all_metrics = {"rewards": [], "steps": [], "successes": [], "epsilons": []}

    for i, seed in enumerate(seeds):
        env_cfg = {**ENV_CONFIG, "seed": seed}
        env = MazeEnv(**env_cfg)

        print(f"\n--- Maze {i + 1}/{len(seeds)} (seed={seed}) ---")
        metrics = train(env, agent, episodes=episodes_per_maze)

        for key in all_metrics:
            all_metrics[key].extend(metrics[key])

    print("\n" + "-" * 60)
    print("Evaluating on each maze:")
    for seed in seeds:
        env = MazeEnv(**{**ENV_CONFIG, "seed": seed})
        print(f"\n  Maze seed={seed}:")
        evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "q_table_multi.pkl"))
    plot_metrics(all_metrics, save_path=os.path.join(OUTPUT_DIR, "multi_maze_metrics.png"))

    print("\n" + "-" * 60)
    print("Visualizing on last trained maze...")
    last_env = MazeEnv(**{**ENV_CONFIG, "seed": seeds[-1]})
    visualize_single(last_env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "q_learning_multi.gif"))

    return agent, all_metrics



if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    save_config(os.path.join(OUTPUT_DIR, "config.json"))

    single_agent, single_train, single_eval = train_single_maze()
    multi_agent, multi_metrics = train_multi_maze()

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print(f"  Files: config.json, q_table_single.pkl, q_table_multi.pkl,")
    print(f"         single_maze_metrics.png, multi_maze_metrics.png")