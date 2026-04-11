"""
Training Script for Deep Q-Network (DQN) Agent

Single entry point to configure, train, evaluate, and save the DQN agent.
Run: python training/dqn_training.py
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from envs.maze_env import MazeEnv
from agents.dqn_agent import DQNAgent
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
    "lr": 5e-4,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.999,
    "batch_size": 256,
    "buffer_capacity": 500_000,
    "min_replay_size": 2_000,
    "target_update_freq": 500,
    "hidden_sizes": (512, 256),
    "learn_every": 4,           
    "device": "cpu",  
}

TRAIN_CONFIG = {
    "episodes": 3000,
    "eval_episodes": 100,
    "multi_maze_seeds": [42, 99, 7, 123, 256],
    "cycle_episodes": 100,  
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "dqn")


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_config(path):
    """Save all configs to a JSON file for reproducibility."""
    config = {
        "env": ENV_CONFIG,
        "agent": {k: list(v) if isinstance(v, tuple) else v for k, v in AGENT_CONFIG.items()},
        "training": TRAIN_CONFIG,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def train(env, agent, episodes, maze_label=""):
    """
    Train the DQN agent on the given environment.

    Args:
        env: Gymnasium-compatible MazeEnv instance.
        agent: DQNAgent instance.
        episodes: Number of training episodes.
        maze_label: Optional prefix for progress output.

    Returns:
        Dictionary with keys: rewards, steps, successes, epsilons, losses.
    """
    metrics = {"rewards": [], "steps": [], "successes": [], "epsilons": [], "losses": []}
    window_start = time.perf_counter()

    for episode in range(episodes):
        obs, _ = env.reset()
        state_vec    = agent._encode(tuple(obs))   # encode once; reused each step
        total_reward = 0.0
        ep_losses    = []
        done  = False
        steps = 0

        while not done:
            action = agent.choose_action_vec(state_vec)  # no re-encode
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state_vec = agent._encode(tuple(next_obs))
            done = terminated or truncated

            agent.buffer.push(state_vec, action, reward, next_state_vec, float(terminated))

            if steps % agent.learn_every == 0:
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)

            state_vec = next_state_vec  
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(info["steps"])
        metrics["successes"].append(1 if terminated else 0)
        metrics["epsilons"].append(agent.epsilon)
        metrics["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)

        if (episode + 1) % 100 == 0:
            elapsed        = time.perf_counter() - window_start
            recent_reward  = np.mean(metrics["rewards"][-100:])
            recent_success = np.mean(metrics["successes"][-100:]) * 100
            recent_loss    = np.mean(metrics["losses"][-100:])
            print(f"{maze_label}Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {recent_reward:.2f} | "
                  f"Success Rate: {recent_success:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Loss: {recent_loss:.4f} | "
                  f"Time: {elapsed:.1f}s")
            window_start = time.perf_counter()

    return metrics


def evaluate(env, agent, episodes=100):
    """
    Evaluate the trained agent without exploration.

    Args:
        env: Gymnasium-compatible MazeEnv instance.
        agent: Trained DQNAgent instance.
        episodes: Number of evaluation episodes.

    Returns:
        Dictionary with keys: rewards, steps, successes.
    """
    metrics = {"rewards": [], "steps": [], "successes": []}

    for _ in range(episodes):
        obs, _ = env.reset()
        state  = tuple(obs)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_best_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            state = tuple(next_obs)
            total_reward += reward
            done = terminated or truncated

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(info["steps"])
        metrics["successes"].append(1 if terminated else 0)

    avg_reward   = np.mean(metrics["rewards"])
    avg_steps    = np.mean(metrics["steps"])
    success_rate = np.mean(metrics["successes"]) * 100

    print(f"\nEvaluation ({episodes} episodes):")
    print(f"  Avg Reward:   {avg_reward:.2f}")
    print(f"  Avg Steps:    {avg_steps:.1f}")
    print(f"  Success Rate: {success_rate:.1f}%")

    return metrics


def plot_metrics(metrics, save_path=None, title="DQN Training Metrics",
                 maze_boundaries=None):
    """
    Plot training curves: reward, steps, success rate, epsilon, and loss.

    Args:
        metrics: Dictionary from the train() function.
        save_path: Optional path to save the plot image.
        title: Main title for the plot.
        maze_boundaries: Optional list of (episode_index, label) tuples
                         to draw vertical lines separating different mazes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    window = 50

    rewards = np.array(metrics["rewards"])
    rolling_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0, 0].plot(rolling_rewards, color="steelblue")
    axes[0, 0].set_title("Cumulative Reward Over Training")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Avg Reward (per episode)")

    steps = np.array(metrics["steps"])
    rolling_steps = np.convolve(steps, np.ones(window) / window, mode="valid")
    axes[0, 1].plot(rolling_steps, color="coral")
    axes[0, 1].set_title("Steps to Reach Exit")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Avg Steps (per episode)")

    successes = np.array(metrics["successes"])
    rolling_success = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
    axes[0, 2].plot(rolling_success, color="seagreen")
    axes[0, 2].set_title("Maze Solve Rate")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Success %")

    axes[1, 0].plot(metrics["epsilons"], color="mediumpurple")
    axes[1, 0].set_title("Exploration vs Exploitation")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon (exploration rate)")

    losses = np.array(metrics["losses"])
    nonzero = losses[losses > 0]
    if len(nonzero) > 0:
        rolling_loss = np.convolve(nonzero, np.ones(window) / window, mode="valid")
        axes[1, 1].plot(rolling_loss, color="darkorange")
    axes[1, 1].set_title("Training Loss (Huber)")
    axes[1, 1].set_xlabel("Episode (loss computed)")
    axes[1, 1].set_ylabel("Avg Loss")

    axes[1, 2].axis("off")

    if maze_boundaries:
        # With frequent cycling there can be many boundaries; only label the
        # first maze of each cycle (label == "M1") to keep the chart readable.
        for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]:
            for ep, label in maze_boundaries:
                is_cycle_start = label == "M1"
                ax.axvline(x=ep, color="gray", linestyle="--",
                           linewidth=1.2 if is_cycle_start else 0.4,
                           alpha=0.8 if is_cycle_start else 0.3)
                if is_cycle_start:
                    cycle_num = ep // (len(TRAIN_CONFIG["multi_maze_seeds"])
                                       * TRAIN_CONFIG["cycle_episodes"]) + 1
                    ax.text(ep, ax.get_ylim()[1], f" C{cycle_num}", fontsize=8,
                            va="top", ha="left", color="gray", fontweight="bold")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to {save_path}")

    plt.show()


def train_single_maze():
    """Train, evaluate, and visualize the DQN agent on a single maze."""
    print("=" * 60)
    print("  DQN — Single Maze Training")
    print("=" * 60)

    env = MazeEnv(**ENV_CONFIG)
    agent = DQNAgent(maze_grid=env.grid, **AGENT_CONFIG)

    print(f"State size: {agent.state_size}  |  Device: {agent.device}\n")

    train_metrics = train(env, agent, episodes=TRAIN_CONFIG["episodes"])

    print("\n" + "-" * 60)
    evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "dqn_single.pt"))
    plot_metrics(train_metrics,
                 save_path=os.path.join(OUTPUT_DIR, "single_maze_metrics.png"),
                 title="DQN Training Metrics (Single Maze)")

    print("\n" + "-" * 60)
    print("Visualizing trained agent...")
    visualize_single(env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "dqn_single.gif"),
                     title="DQN")

    return agent, train_metrics


def train_multi_maze(agent=None):
    """Train the DQN agent across multiple mazes and visualize on the last one.

    Args:
        agent: Optional pre-trained DQNAgent to warm-start from (e.g. from
               train_single_maze). If None a fresh agent is created.
    """
    print("\n" + "=" * 60)
    print("  DQN — Multi-Maze Training")
    print("=" * 60)

    seeds         = TRAIN_CONFIG["multi_maze_seeds"]
    cycle_eps     = TRAIN_CONFIG["cycle_episodes"]
    total_eps     = TRAIN_CONFIG["episodes"]
    n_cycles      = total_eps // (len(seeds) * cycle_eps)

    envs = [MazeEnv(**{**ENV_CONFIG, "seed": s}) for s in seeds]

    if agent is None:
        agent = DQNAgent(maze_grid=envs[0].grid, **AGENT_CONFIG)
        print("Starting multi-maze from scratch.")
    else:
        agent.set_maze(envs[0].grid)
        print("Warm-starting multi-maze from single-maze trained model.")

    print(f"State size: {agent.state_size}  |  Device: {agent.device}")
    print(f"Schedule: {n_cycles} cycles × {len(seeds)} mazes × {cycle_eps} eps "
          f"= {n_cycles * len(seeds) * cycle_eps} total episodes\n")

    all_metrics    = {"rewards": [], "steps": [], "successes": [], "epsilons": [], "losses": []}
    maze_boundaries = []
    ep_counter     = 0

    for cycle in range(n_cycles):
        print(f"\n=== Cycle {cycle + 1}/{n_cycles} ===")
        for i, env in enumerate(envs):
            agent.set_maze(env.grid)
            metrics = train(env, agent, episodes=cycle_eps,
                            maze_label=f"[C{cycle + 1} M{i + 1}] ")
            for key in all_metrics:
                all_metrics[key].extend(metrics[key])
            maze_boundaries.append((ep_counter, f"M{i + 1}"))
            ep_counter += cycle_eps

    last_env = envs[-1]

    print("\n" + "-" * 60)
    print("Evaluating on each maze:")
    for seed in seeds:
        env = MazeEnv(**{**ENV_CONFIG, "seed": seed})
        agent.set_maze(env.grid)
        print(f"\n  Maze seed={seed}:")
        evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "dqn_multi.pt"))
    plot_metrics(all_metrics,
                 save_path=os.path.join(OUTPUT_DIR, "multi_maze_metrics.png"),
                 title="DQN Training Metrics (Multi-Maze)",
                 maze_boundaries=maze_boundaries)

    agent.set_maze(last_env.grid)
    print("\n" + "-" * 60)
    print("Visualizing trained agent on last maze...")
    visualize_single(last_env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "dqn_multi.gif"),
                     title="DQN")

    return agent, all_metrics


def test():
    """Load saved models and run evaluation + visualisation (no training)."""
    single_path = os.path.join(OUTPUT_DIR, "dqn_single.pt")
    multi_path  = os.path.join(OUTPUT_DIR, "dqn_multi.pt")

    for label, ckpt_path in [("Single", single_path), ("Multi", multi_path)]:
        if not os.path.exists(ckpt_path):
            print(f"[{label}] No saved model found at {ckpt_path} — skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"  DQN — {label} Maze Evaluation")
        print("=" * 60)

        env   = MazeEnv(**ENV_CONFIG)
        agent = DQNAgent(maze_grid=env.grid, **AGENT_CONFIG)
        agent.load(ckpt_path)
        agent.epsilon = 0.0 

        print(f"Loaded {ckpt_path}")
        evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

        gif_path = os.path.join(OUTPUT_DIR, f"dqn_{label.lower()}_test.gif")
        visualize_single(env, agent, save_path=gif_path, title=f"DQN ({label})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Maze Solver")
    parser.add_argument("--train", action="store_true",
                        help="Retrain from scratch. Omit to run in test/eval mode.")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    if args.train:
        save_config(os.path.join(OUTPUT_DIR, "config.json"))
        agent, _ = train_single_maze()
        train_multi_maze()

        print("\n" + "=" * 60)
        print("  Training Complete")
        print("=" * 60)
        print(f"  Outputs saved to: {OUTPUT_DIR}")
    else:
        test()
