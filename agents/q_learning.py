"""
Q-Learning Agent for MazeRL Project

Tabular Q-Learning with epsilon-greedy exploration and decaying epsilon.
Serves as the baseline RL agent for comparison against DQN and PPO.
"""

import numpy as np
import pickle
from collections import defaultdict


class QLearningAgent:
    """
    Tabular Q-Learning agent for discrete state-action maze environments.

    Args:
        action_size: Number of possible actions.
        learning_rate: Step size for Q-value updates (alpha).
        discount_factor: Weight for future rewards (gamma).
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay applied per episode.
    """

    def __init__(self, action_size=4, learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def choose_action(self, state):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state as a tuple (row, col).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value for the given state-action pair.

        Args:
            state: Current state tuple.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state tuple.
            done: Whether the episode ended.
        """
        current_q = self.q_table[state][action]
        next_q = 0.0 if done else np.max(self.q_table[next_state])
        target = reward + self.gamma * next_q
        self.q_table[state][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_best_action(self, state):
        """Return the greedy action (no exploration) for a given state."""
        return int(np.argmax(self.q_table[state]))

    def save(self, filepath):
        """
        Save the Q-table to a file.

        Args:
            filepath: Path to save the pickle file.
        """
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        """
        Load a Q-table from a file.

        Args:
            filepath: Path to the pickle file.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data)


def train(env, agent, episodes=1000):
    """
    Train the Q-Learning agent on the given environment.

    Args:
        env: Gymnasium-compatible maze environment.
        agent: QLearningAgent instance.
        episodes: Number of training episodes.

    Returns:
        Dictionary containing training metrics (rewards, steps, successes, epsilons).
    """
    metrics = {"rewards": [], "steps": [], "successes": [], "epsilons": []}

    for episode in range(episodes):
        obs, info = env.reset()
        state = tuple(obs)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = tuple(next_obs)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(info["steps"])
        metrics["successes"].append(1 if terminated else 0)
        metrics["epsilons"].append(agent.epsilon)

        if (episode + 1) % 100 == 0:
            recent_rewards = np.mean(metrics["rewards"][-100:])
            recent_success = np.mean(metrics["successes"][-100:]) * 100
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {recent_rewards:.2f} | "
                  f"Success Rate: {recent_success:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return metrics


def evaluate(env, agent, episodes=100):
    """
    Evaluate the trained agent without exploration.

    Args:
        env: Gymnasium-compatible maze environment.
        agent: Trained QLearningAgent instance.
        episodes: Number of evaluation episodes.

    Returns:
        Dictionary containing evaluation metrics.
    """
    metrics = {"rewards": [], "steps": [], "successes": []}

    for _ in range(episodes):
        obs, info = env.reset()
        state = tuple(obs)
        total_reward = 0
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

    avg_reward = np.mean(metrics["rewards"])
    avg_steps = np.mean(metrics["steps"])
    success_rate = np.mean(metrics["successes"]) * 100

    print(f"\nEvaluation ({episodes} episodes):")
    print(f"  Avg Reward:   {avg_reward:.2f}")
    print(f"  Avg Steps:    {avg_steps:.1f}")
    print(f"  Success Rate: {success_rate:.1f}%")

    return metrics


def plot_metrics(metrics, save_path=None, title="Q-Learning Training Metrics",
                 maze_boundaries=None):
    """
    Plot training curves: reward, steps, success rate, and epsilon.

    Args:
        metrics: Dictionary from the train() function.
        save_path: Optional path to save the plot image.
        title: Main title for the plot.
        maze_boundaries: Optional list of (episode_index, label) tuples
                         to draw vertical lines separating different mazes.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
    axes[1, 0].plot(rolling_success, color="seagreen")
    axes[1, 0].set_title("Maze Solve Rate")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Success %")

    axes[1, 1].plot(metrics["epsilons"], color="mediumpurple")
    axes[1, 1].set_title("Exploration vs Exploitation")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Epsilon (exploration rate)")

    if maze_boundaries:
        for ax in axes.flat:
            for ep, label in maze_boundaries:
                ax.axvline(x=ep, color="gray", linestyle="--", linewidth=1, alpha=0.7)
                ax.text(ep, ax.get_ylim()[1], f" {label}", fontsize=8,
                        va="top", ha="left", color="gray", fontweight="bold")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to {save_path}")

    plt.show()
