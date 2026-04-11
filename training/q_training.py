"""
Training Script for Q-Learning Agent

Single entry point to configure, train, evaluate, and save the Q-Learning agent.
Run: python training/q_training.py          (test/eval mode)
     python training/q_training.py --train  (retrain from scratch)
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs.maze_env import MazeEnv
from agents.q_learning import QLearningAgent, train, evaluate, plot_metrics
from utils.visualization import visualize_single


ENV_CONFIG = {
    "width": 11,
    "height": 11,
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


def train_multi_maze(agent=None):
    """Train across multiple mazes and visualize on the last one.

    Args:
        agent: Optional pre-trained QLearningAgent to warm-start from (e.g.
               from train_single_maze). If None a fresh agent is created.
    """
    print("\n" + "=" * 60)
    print("  Q-Learning — Multi-Maze Training")
    print("=" * 60)

    seeds = TRAIN_CONFIG["multi_maze_seeds"]
    episodes_per_maze = TRAIN_CONFIG["episodes"] // len(seeds)

    if agent is None:
        agent = QLearningAgent(**AGENT_CONFIG)
        print("Starting multi-maze from scratch.")
    else:
        print("Warm-starting multi-maze from single-maze trained model.")

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


def test():
    """Load saved Q-tables and run evaluation + visualisation (no training)."""
    checkpoints = [
        ("Single", os.path.join(OUTPUT_DIR, "q_table_single.pkl")),
        ("Multi",  os.path.join(OUTPUT_DIR, "q_table_multi.pkl")),
    ]

    for label, ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"[{label}] No saved model found at {ckpt_path} — skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"  Q-Learning — {label} Maze Evaluation")
        print("=" * 60)

        env = MazeEnv(**ENV_CONFIG)
        agent = QLearningAgent(**AGENT_CONFIG)
        agent.load(ckpt_path)
        agent.epsilon = 0.0     # pure greedy for evaluation

        print(f"Loaded {ckpt_path}")
        evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

        gif_path = os.path.join(OUTPUT_DIR, f"q_learning_{label.lower()}_test.gif")
        visualize_single(env, agent, save_path=gif_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Learning Maze Solver")
    parser.add_argument("--train", action="store_true",
                        help="Retrain from scratch. Omit to run in test/eval mode.")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    if args.train:
        save_config(os.path.join(OUTPUT_DIR, "config.json"))
        # Single-maze training; returned agent warm-starts multi-maze training.
        agent, _ = train_single_maze()
        train_multi_maze(agent=agent)

        print("\n" + "=" * 60)
        print("  Training Complete")
        print("=" * 60)
        print(f"  Outputs saved to: {OUTPUT_DIR}")
    else:
        test()
