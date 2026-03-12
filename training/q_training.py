"""
Training Script for Q-Learning Agent

Single entry point to configure, train, evaluate, and save the Q-Learning agent.
Run: python training/train_qlearning.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze_env import MazeEnv
from agents.q_learning import QLearningAgent, train, evaluate, plot_metrics
from utils.visualization import visualize_single


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
    "episodes": 10000,
    "eval_episodes": 100,
    "multi_maze_seeds": [42, 99, 7, 123, 256],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "q_learning")


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


def train_single_maze():
    """Train, evaluate, and visualize on a single maze."""
    print("=" * 60)
    print("  Q-Learning — Single Maze Training")
    print("=" * 60)

    env = MazeEnv(**ENV_CONFIG)
    agent = QLearningAgent(**AGENT_CONFIG)

    train_metrics = train(env, agent, episodes=TRAIN_CONFIG["episodes"])

    print("\n" + "-" * 60)
    evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "q_table_single.pkl"))
    plot_metrics(train_metrics,
                save_path=os.path.join(OUTPUT_DIR, "single_maze_metrics.png"),
                title="Q-Learning Training Metrics (Single Maze)")

    print("\n" + "-" * 60)
    print("Visualizing trained agent...")
    visualize_single(env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "q_learning_single.gif"))

    return agent, train_metrics


def train_multi_maze():
    """Train across multiple mazes and visualize on the last one."""
    print("\n" + "=" * 60)
    print("  Q-Learning — Multi-Maze Training")
    print("=" * 60)

    seeds = TRAIN_CONFIG["multi_maze_seeds"]
    episodes_per_maze = TRAIN_CONFIG["episodes"] // len(seeds)
    agent = QLearningAgent(**AGENT_CONFIG)

    all_metrics = {"rewards": [], "steps": [], "successes": [], "epsilons": []}
    maze_boundaries = []

    for i, seed in enumerate(seeds):
        env = MazeEnv(**{**ENV_CONFIG, "seed": seed})
        print(f"\n--- Maze {i + 1}/{len(seeds)} (seed={seed}) ---")
        metrics = train(env, agent, episodes=episodes_per_maze)
        for key in all_metrics:
            all_metrics[key].extend(metrics[key])
        maze_boundaries.append((i * episodes_per_maze, f"Maze {i + 1}"))

    print("\n" + "-" * 60)
    print("Evaluating on each maze:")
    for seed in seeds:
        env = MazeEnv(**{**ENV_CONFIG, "seed": seed})
        print(f"\n  Maze seed={seed}:")
        evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "q_table_multi.pkl"))
    plot_metrics(all_metrics,
                save_path=os.path.join(OUTPUT_DIR, "multi_maze_metrics.png"),
                title="Q-Learning Training Metrics (Multi-Maze)",
                maze_boundaries=maze_boundaries)

    return agent, all_metrics



if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    save_config(os.path.join(OUTPUT_DIR, "config.json"))

    train_single_maze()
    train_multi_maze()

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Outputs saved to: {OUTPUT_DIR}")