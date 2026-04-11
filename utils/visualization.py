"""
Maze Visualizer for MazeRL Project

Animates a trained agent solving the maze step by step using Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.animation as animation


CELL_COLORS = mcolors.ListedColormap(["white", "black", "green", "red", "orange"])
CELL_BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
CELL_NORM = mcolors.BoundaryNorm(CELL_BOUNDS, CELL_COLORS.N)

LEGEND = [
    Patch(facecolor="white", edgecolor="black", label="Path"),
    Patch(facecolor="black", edgecolor="black", label="Wall"),
    Patch(facecolor="green", edgecolor="black", label="Start"),
    Patch(facecolor="red", edgecolor="black", label="Exit"),
    Patch(facecolor="orange", edgecolor="black", label="Trap"),
]


def record_agent_path(env, agent, max_steps=500, stochastic=False):
    """
    Run the trained agent through the maze and record its path.

    Args:
        env: MazeEnv instance.
        agent: Trained agent instance.
        max_steps: Safety limit.
        stochastic: If True, sample from the policy distribution instead of
                    taking the argmax. Use this for PPO agents to avoid
                    deterministic cycles.

    Returns:
        List of (row, col) positions the agent visited.
    """
    obs, _ = env.reset()
    state = tuple(obs)
    path = [state]
    visited = {state: 0}  # state -> first step seen (cycle detection)

    for step in range(max_steps):
        if stochastic and hasattr(agent, "choose_action"):
            action, _, _ = agent.choose_action(state)
        else:
            action = agent.get_best_action(state)

        obs, _, terminated, truncated, _ = env.step(action)
        state = tuple(obs)
        path.append(state)

        if terminated or truncated:
            break

        # Cycle detection: if we revisit a state we've seen recently (within
        # the last 20 steps), the deterministic policy is looping — stop early.
        if not stochastic and state in visited and (step - visited[state]) <= 20:
            break
        visited[state] = step

    return path


def draw_maze(ax, grid, height, width):
    """Draw the base maze grid on the given axes."""
    ax.imshow(grid, cmap=CELL_COLORS, norm=CELL_NORM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for x in range(width + 1):
        ax.axvline(x - 0.5, color="gray", linewidth=0.3)
    for y in range(height + 1):
        ax.axhline(y - 0.5, color="gray", linewidth=0.3)


def visualize_single(env, agent, save_path=None, interval=150, title="Q-Learning",
                     stochastic=False):
    """
    Animate a single agent solving the maze.

    Args:
        env: MazeEnv instance.
        agent: Trained agent instance with a get_best_action(state) method.
        save_path: Optional path to save as .gif or .mp4.
        interval: Milliseconds between frames.
        title: Label shown in the animation title bar.
    """
    path = record_agent_path(env, agent, stochastic=stochastic)
    grid = env.grid
    height, width = grid.shape
    reached_exit = (path[-1] == env.exit_pos)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_maze(ax, grid, height, width)
    ax.legend(handles=LEGEND, loc="upper right", fontsize=9, framealpha=0.9)

    trail, = ax.plot([], [], color="dodgerblue", linewidth=2, alpha=0.5)
    agent_dot, = ax.plot([], [], "bo", markersize=14,
                         markeredgecolor="darkblue", markeredgewidth=2)
    title_text = ax.set_title("", fontsize=14, fontweight="bold")

    def update(frame):
        current_path = path[:frame + 1]
        cols = [p[1] for p in current_path]
        rows = [p[0] for p in current_path]

        trail.set_data(cols, rows)
        agent_dot.set_data([cols[-1]], [rows[-1]])

        status = "SOLVED!" if frame == len(path) - 1 and reached_exit else ""
        title_text.set_text(f"{title}  |  Step: {frame}/{len(path)-1}  {status}")
        return trail, agent_dot, title_text

    anim = animation.FuncAnimation(
        fig, update, frames=len(path), interval=interval, blit=True, repeat=False
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)