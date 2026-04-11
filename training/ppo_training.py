"""
Training Script for Proximal Policy Optimization (PPO) Agent

Single entry point to configure, train, evaluate, and save the PPO agent.
Run: python training/ppo_training.py [--train]
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
from agents.ppo_agent import PPOAgent
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
    "lr": 2e-4,           
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.05, 
    "max_grad_norm": 0.5,
    "n_steps": 2048,
    "ppo_epochs": 10,    
    "mini_batch_size": 64,
    "hidden_sizes": (256, 128),
    "target_kl": 0.02,
    "device": "cpu",
}

TRAIN_CONFIG = {
    "total_timesteps": 600_000,
    "eval_episodes": 100,
    "multi_maze_seeds": [42, 99, 7, 123, 256],
    "cycle_timesteps": 60_000
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "ppo")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_config(path):
    config = {
        "env": ENV_CONFIG,
        "agent": {k: list(v) if isinstance(v, tuple) else v for k, v in AGENT_CONFIG.items()},
        "training": TRAIN_CONFIG,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def train(env, agent, total_timesteps, maze_label="",
          patience=5, min_delta=0.05, hard_stop_delta=0.15,
          min_episodes=100, best_ckpt_path=None):
    """
    Train the PPO agent with early stopping.

    Collects n_steps transitions into the rollout buffer, runs a PPO update,
    then repeats until total_timesteps are consumed or early stopping fires.

    Two-tier early stopping:
      1. Hard stop  — if success drops >= hard_stop_delta in a single check,
                      stop immediately and restore the best checkpoint.
      2. Soft stop  — if success stays more than min_delta below the best for
                      `patience` consecutive updates, stop and restore.

    Best checkpoint is saved to disk (best_ckpt_path) every time a new best
    is found, then loaded back at the end — avoids any in-memory copy issues.

    Args:
        env: Gymnasium-compatible MazeEnv instance.
        agent: PPOAgent instance.
        total_timesteps: Total environment interaction steps for this call.
        maze_label: Optional prefix for progress output.
        patience: Consecutive updates below best before soft stop triggers.
        min_delta: Drop threshold that increments the patience counter.
        hard_stop_delta: Single-check drop that triggers an immediate stop.
        min_episodes: Minimum completed episodes before early stopping activates.
        best_ckpt_path: File path to save the best checkpoint. Defaults to a
                        temp file inside OUTPUT_DIR.

    Returns:
        Dictionary with keys: rewards, steps, successes,
                              actor_losses, critic_losses, entropies.
    """
    if best_ckpt_path is None:
        best_ckpt_path = os.path.join(OUTPUT_DIR, "_best_ckpt_tmp.pt")

    metrics = {
        "rewards": [], "steps": [], "successes": [],
        "actor_losses": [], "critic_losses": [], "entropies": [],
    }

    obs, _    = env.reset()
    state_vec = agent._encode(tuple(obs))
    ep_reward = 0.0
    ep_count  = 0
    timestep  = 0
    done      = False
    window_start  = time.perf_counter()
    report_every  = 50

    best_success = -1.0
    best_reward  = -np.inf
    no_improve   = 0
    stopped_early = False

    while timestep < total_timesteps:
        action, log_prob, value = agent.choose_action_vec(state_vec)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_vec = agent._encode(tuple(next_obs))

        agent.store(state_vec, action, reward, done, log_prob, value)

        ep_reward += reward
        timestep  += 1

        if done:
            metrics["rewards"].append(ep_reward)
            metrics["steps"].append(info["steps"])
            metrics["successes"].append(1 if terminated else 0)
            ep_count += 1

            if ep_count % report_every == 0:
                elapsed        = time.perf_counter() - window_start
                recent_reward  = np.mean(metrics["rewards"][-report_every:])
                recent_success = np.mean(metrics["successes"][-report_every:]) * 100
                recent_loss    = np.mean(metrics["actor_losses"][-5:]) if metrics["actor_losses"] else 0.0
                print(f"{maze_label}Episode {ep_count} (step {timestep}/{total_timesteps}) | "
                      f"Avg Reward: {recent_reward:.2f} | "
                      f"Success Rate: {recent_success:.1f}% | "
                      f"Actor Loss: {recent_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")
                window_start = time.perf_counter()

            obs, _    = env.reset()
            state_vec = agent._encode(tuple(obs))
            ep_reward = 0.0
        else:
            state_vec = next_state_vec

        if agent.buffer_full():

            last_value  = 0.0 if done else agent.get_value(state_vec)
            update_info = agent.update(last_value)
            metrics["actor_losses"].append(update_info["actor_loss"])
            metrics["critic_losses"].append(update_info["critic_loss"])
            metrics["entropies"].append(update_info["entropy"])

            if ep_count >= min_episodes:
                window      = min(100, ep_count)
                cur_success = np.mean(metrics["successes"][-window:])
                cur_reward  = np.mean(metrics["rewards"][-window:])

               
                is_better = (cur_success > best_success or
                             (cur_success == best_success and cur_reward > best_reward))

                if is_better:
                    best_success = cur_success
                    best_reward  = cur_reward
                    agent.save(best_ckpt_path) 
                    no_improve   = 0
                    print(f"{maze_label} New best: success={best_success:.1%}  reward={best_reward:.2f}  (step {timestep})")

                elif cur_success <= best_success - hard_stop_delta:
                    stopped_early = True
                    print(f"\n{maze_label}Hard stop at step {timestep} "
                          f"(success dropped {best_success:.1%} → {cur_success:.1%})")
                    break

                elif cur_success < best_success - min_delta:
                    no_improve += 1
                    if no_improve >= patience:
                        stopped_early = True
                        print(f"\n{maze_label}Early stopping at step {timestep} "
                              f"(best: {best_success:.1%}, current: {cur_success:.1%}, "
                              f"{patience} updates without improvement)")
                        break

                else:
                    no_improve = 0 

    if os.path.exists(best_ckpt_path):
        agent.load(best_ckpt_path)
        if stopped_early:
            print(f"{maze_label}Restored best checkpoint (success={best_success:.1%}, reward={best_reward:.2f})")

    return metrics


def evaluate(env, agent, episodes=100):
    """
    Evaluate the trained agent with the greedy (argmax) policy.

    Args:
        env: Gymnasium-compatible MazeEnv instance.
        agent: Trained PPOAgent instance.
        episodes: Number of evaluation episodes.

    Returns:
        Dictionary with keys: rewards, steps, successes.
    """
    metrics = {"rewards": [], "steps": [], "successes": []}

    for _ in range(episodes):
        obs, _       = env.reset()
        state        = tuple(obs)
        total_reward = 0.0
        done         = False

        while not done:
         
            action, _, _ = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            state        = tuple(next_obs)
            total_reward += reward
            done         = terminated or truncated

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


def plot_metrics(metrics, save_path=None, title="PPO Training Metrics",
                 maze_boundaries=None):
    """
    Plot training curves: reward, steps, success rate, actor loss, critic loss, entropy.

    Args:
        metrics: Dictionary from the train() function.
        save_path: Optional path to save the plot image.
        title: Main title for the plot.
        maze_boundaries: Optional list of (episode_index, label) tuples
                         to draw vertical separators between mazes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    window = 50

    def rolling(arr):
        return np.convolve(arr, np.ones(window) / window, mode="valid") if len(arr) >= window else arr

    rewards = np.array(metrics["rewards"])
    axes[0, 0].plot(rolling(rewards), color="steelblue")
    axes[0, 0].set_title("Cumulative Reward Over Training")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Avg Reward (per episode)")

    steps = np.array(metrics["steps"])
    axes[0, 1].plot(rolling(steps), color="coral")
    axes[0, 1].set_title("Steps to Reach Exit")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Avg Steps (per episode)")

    successes = np.array(metrics["successes"])
    axes[0, 2].plot(rolling(successes) * 100, color="seagreen")
    axes[0, 2].set_title("Maze Solve Rate")
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Success %")

    actor_losses = np.array(metrics["actor_losses"])
    if len(actor_losses) > 0:
        axes[1, 0].plot(actor_losses, color="mediumpurple")
    axes[1, 0].set_title("Actor Loss (PPO-Clip)")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].set_ylabel("Actor Loss")

    critic_losses = np.array(metrics["critic_losses"])
    if len(critic_losses) > 0:
        axes[1, 1].plot(critic_losses, color="darkorange")
    axes[1, 1].set_title("Critic Loss (Huber)")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("Critic Loss")

    entropies = np.array(metrics["entropies"])
    if len(entropies) > 0:
        axes[1, 2].plot(entropies, color="teal")
    axes[1, 2].set_title("Policy Entropy")
    axes[1, 2].set_xlabel("Update")
    axes[1, 2].set_ylabel("Entropy")

    if maze_boundaries:
        for ax in [axes[0, 0], axes[0, 1], axes[0, 2]]:
            for ep, label in maze_boundaries:
                is_cycle_start = label == "M1"
                ax.axvline(x=ep, color="gray", linestyle="--",
                           linewidth=1.2 if is_cycle_start else 0.4,
                           alpha=0.8 if is_cycle_start else 0.3)
                if is_cycle_start:
                    cycle_num = ep // (len(TRAIN_CONFIG["multi_maze_seeds"])
                                       * TRAIN_CONFIG["cycle_timesteps"]) + 1
                    ax.text(ep, ax.get_ylim()[1], f" C{cycle_num}", fontsize=8,
                            va="top", ha="left", color="gray", fontweight="bold")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to {save_path}")

    plt.show()


def train_single_maze():
    """Train, evaluate, and visualize the PPO agent on a single maze."""
    print("=" * 60)
    print("  PPO — Single Maze Training")
    print("=" * 60)

    env   = MazeEnv(**ENV_CONFIG)
    agent = PPOAgent(maze_grid=env.grid, **AGENT_CONFIG)

    print(f"State size: {agent.state_size}  |  Device: {agent.device}\n")

    train_metrics = train(env, agent, total_timesteps=TRAIN_CONFIG["total_timesteps"])

    print("\n" + "-" * 60)
    evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "ppo_single.pt"))
    plot_metrics(train_metrics,
                 save_path=os.path.join(OUTPUT_DIR, "single_maze_metrics.png"),
                 title="PPO Training Metrics (Single Maze)")

    print("\n" + "-" * 60)
    print("Visualizing trained agent...")
    visualize_single(env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "ppo_single.gif"),
                     title="PPO", stochastic=True)

    return agent, train_metrics


def train_multi_maze(agent=None):
    """Train the PPO agent across multiple mazes using random maze selection.

    At every episode reset a maze is chosen uniformly at random from the pool.
    This ensures all mazes are represented within every rollout, preventing the
    catastrophic forgetting that occurs when training on one maze for long blocks.

    Args:
        agent: Optional pre-trained PPOAgent to warm-start from.
               If None, a fresh agent is created.
    """
    print("\n" + "=" * 60)
    print("  PPO — Multi-Maze Training (randomised)")
    print("=" * 60)

    seeds    = TRAIN_CONFIG["multi_maze_seeds"]
    total_ts = TRAIN_CONFIG["total_timesteps"]
    envs     = [MazeEnv(**{**ENV_CONFIG, "seed": s}) for s in seeds]

    if agent is None:
        agent = PPOAgent(maze_grid=envs[0].grid, **AGENT_CONFIG)
        print("Starting multi-maze from scratch.")
    else:
        agent.set_maze(envs[0].grid)
        print("Warm-starting multi-maze from single-maze trained model.")

    print(f"State size: {agent.state_size}  |  Device: {agent.device}")
    print(f"Mazes: {len(envs)}  |  Total timesteps: {total_ts}\n")

    metrics = {
        "rewards": [], "steps": [], "successes": [],
        "actor_losses": [], "critic_losses": [], "entropies": [],
    }

    cur_env   = np.random.choice(envs)
    agent.set_maze(cur_env.grid)
    obs, _    = cur_env.reset()
    state_vec = agent._encode(tuple(obs))
    ep_reward = 0.0
    ep_count  = 0
    timestep  = 0
    done      = False
    window_start = time.perf_counter()
    report_every = 50

    best_ckpt_path = os.path.join(OUTPUT_DIR, "_best_ckpt_tmp.pt")
    best_success   = -1.0
    best_reward    = -np.inf
    no_improve     = 0
    patience       = 30         
    min_delta      = 0.10       
    hard_stop_delta = 0.30       
    min_episodes   = 1000       

    while timestep < total_ts:
        action, log_prob, value = agent.choose_action_vec(state_vec)
        next_obs, reward, terminated, truncated, info = cur_env.step(action)
        done = terminated or truncated
        next_state_vec = agent._encode(tuple(next_obs))

        agent.store(state_vec, action, reward, done, log_prob, value)
        ep_reward += reward
        timestep  += 1

        if done:
            metrics["rewards"].append(ep_reward)
            metrics["steps"].append(info["steps"])
            metrics["successes"].append(1 if terminated else 0)
            ep_count += 1

            if ep_count % report_every == 0:
                elapsed        = time.perf_counter() - window_start
                recent_reward  = np.mean(metrics["rewards"][-report_every:])
                recent_success = np.mean(metrics["successes"][-report_every:]) * 100
                recent_loss    = np.mean(metrics["actor_losses"][-5:]) if metrics["actor_losses"] else 0.0
                print(f"Episode {ep_count} (step {timestep}/{total_ts}) | "
                      f"Avg Reward: {recent_reward:.2f} | "
                      f"Success Rate: {recent_success:.1f}% | "
                      f"Actor Loss: {recent_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")
                window_start = time.perf_counter()

            cur_env = np.random.choice(envs)
            agent.set_maze(cur_env.grid)
            obs, _    = cur_env.reset()
            state_vec = agent._encode(tuple(obs))
            ep_reward = 0.0
        else:
            state_vec = next_state_vec

        if agent.buffer_full():
            last_value  = 0.0 if done else agent.get_value(state_vec)
            update_info = agent.update(last_value)
            metrics["actor_losses"].append(update_info["actor_loss"])
            metrics["critic_losses"].append(update_info["critic_loss"])
            metrics["entropies"].append(update_info["entropy"])

            if ep_count >= min_episodes:
                window      = min(200, ep_count)
                cur_success = np.mean(metrics["successes"][-window:])
                cur_reward  = np.mean(metrics["rewards"][-window:])

                is_better = (cur_success > best_success or
                             (cur_success == best_success and cur_reward > best_reward))

                if is_better:
                    best_success = cur_success
                    best_reward  = cur_reward
                    agent.save(best_ckpt_path)
                    no_improve   = 0
                    print(f"  ✓ New best: success={best_success:.1%}  reward={best_reward:.2f}  (step {timestep})")
                elif cur_success <= best_success - hard_stop_delta:
                    print(f"\nHard stop at step {timestep} "
                          f"(success dropped {best_success:.1%} → {cur_success:.1%})")
                    break
                elif cur_success < best_success - min_delta:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"\nEarly stopping at step {timestep} "
                              f"(best: {best_success:.1%}, {patience} updates without improvement)")
                        break
                else:
                    no_improve = 0

    if os.path.exists(best_ckpt_path):
        agent.load(best_ckpt_path)

    print("\n" + "-" * 60)
    print("Evaluating on each maze:")
    for seed in seeds:
        eval_env = MazeEnv(**{**ENV_CONFIG, "seed": seed})
        agent.set_maze(eval_env.grid)
        print(f"\n  Maze seed={seed}:")
        evaluate(eval_env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

    agent.save(os.path.join(OUTPUT_DIR, "ppo_multi.pt"))
    plot_metrics(metrics,
                 save_path=os.path.join(OUTPUT_DIR, "multi_maze_metrics.png"),
                 title="PPO Training Metrics (Multi-Maze, randomised)")

    eval_env = MazeEnv(**{**ENV_CONFIG, "seed": 42})
    agent.set_maze(eval_env.grid)
    print("\n" + "-" * 60)
    print("Visualizing trained agent on maze seed=42...")
    visualize_single(eval_env, agent,
                     save_path=os.path.join(OUTPUT_DIR, "ppo_multi.gif"),
                     title="PPO (Multi)", stochastic=True)

    return agent, metrics


def test():
    """Load saved models and run evaluation + visualisation (no training)."""
    single_path = os.path.join(OUTPUT_DIR, "ppo_single.pt")
    multi_path  = os.path.join(OUTPUT_DIR, "ppo_multi.pt")

    for label, ckpt_path in [("Single", single_path), ("Multi", multi_path)]:
        if not os.path.exists(ckpt_path):
            print(f"[{label}] No saved model found at {ckpt_path} — skipping.")
            continue

        print("\n" + "=" * 60)
        print(f"  PPO — {label} Maze Evaluation")
        print("=" * 60)

        env   = MazeEnv(**ENV_CONFIG)
        agent = PPOAgent(maze_grid=env.grid, **AGENT_CONFIG)
        agent.load(ckpt_path)

        print(f"Loaded {ckpt_path}")
        evaluate(env, agent, episodes=TRAIN_CONFIG["eval_episodes"])

        gif_path = os.path.join(OUTPUT_DIR, f"ppo_{label.lower()}_test.gif")
        visualize_single(env, agent, save_path=gif_path, title=f"PPO ({label})", stochastic=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Maze Solver")
    parser.add_argument("--train", action="store_true",
                        help="Retrain from scratch. Omit to run in test/eval mode.")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    if args.train:
        save_config(os.path.join(OUTPUT_DIR, "config.json"))
        train_single_maze()
    
        train_multi_maze()

        print("\n" + "=" * 60)
        print("  Training Complete")
        print("=" * 60)
        print(f"  Outputs saved to: {OUTPUT_DIR}")
    else:
        test()
