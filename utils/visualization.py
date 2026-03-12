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


def record_agent_path(env, agent, max_steps=500):
    """
    Run the trained agent through the maze and record its path.

    Args:
        env: MazeEnv instance.
        agent: Trained QLearningAgent instance.
        max_steps: Safety limit.

    Returns:
        List of (row, col) positions the agent visited.
    """
    obs, _ = env.reset()
    state = tuple(obs)
    path = [state]

    for _ in range(max_steps):
        action = agent.get_best_action(state)
        obs, _, terminated, truncated, _ = env.step(action)
        state = tuple(obs)
        path.append(state)
        if terminated or truncated:
            break

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


def visualize_single(env, agent, save_path=None, interval=150):
    """
    Animate a single agent solving the maze.

    Args:
        env: MazeEnv instance.
        agent: Trained QLearningAgent instance.
        save_path: Optional path to save as .gif or .mp4.
        interval: Milliseconds between frames.
    """
    path = record_agent_path(env, agent)
    grid = env.grid
    height, width = grid.shape
    reached_exit = (path[-1] == env.exit_pos)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_maze(ax, grid, height, width)
    ax.legend(handles=LEGEND, loc="upper right", fontsize=9, framealpha=0.9)

    trail, = ax.plot([], [], color="dodgerblue", linewidth=2, alpha=0.5)
    agent_dot, = ax.plot([], [], "bo", markersize=14,
                         markeredgecolor="darkblue", markeredgewidth=2)
    title = ax.set_title("", fontsize=14, fontweight="bold")

    def update(frame):
        current_path = path[:frame + 1]
        cols = [p[1] for p in current_path]
        rows = [p[0] for p in current_path]

        trail.set_data(cols, rows)
        agent_dot.set_data([cols[-1]], [rows[-1]])

        status = "SOLVED!" if frame == len(path) - 1 and reached_exit else ""
        title.set_text(f"Q-Learning  |  Step: {frame}/{len(path)-1}  {status}")
        return trail, agent_dot, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(path), interval=interval, blit=True, repeat=False
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def visualize_training_progress(env, snapshots, save_path=None, interval=200):
    """
    Compile snapshots from different training stages into one GIF.
    Shows how the agent improves from random wandering to efficient solving.

    Args:
        env: MazeEnv instance.
        snapshots: List of dicts with keys: episode, path, steps, solved.
        save_path: Optional path to save as .gif.
        interval: Milliseconds between frames.
    """
    grid = env.grid
    height, width = grid.shape

    all_frames = []
    for snap in snapshots:
        path = snap["path"]
        for step_idx in range(len(path)):
            all_frames.append((snap, step_idx))
        for _ in range(8):
            all_frames.append((snap, len(path) - 1))

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_maze(ax, grid, height, width)
    ax.legend(handles=LEGEND, loc="upper right", fontsize=9, framealpha=0.9)

    trail, = ax.plot([], [], linewidth=2, alpha=0.5)
    agent_dot, = ax.plot([], [], "o", markersize=14, markeredgewidth=2)
    title = ax.set_title("", fontsize=14, fontweight="bold")

    def update(frame_idx):
        snap, step_idx = all_frames[frame_idx]
        path = snap["path"]
        current_path = path[:step_idx + 1]

        cols = [p[1] for p in current_path]
        rows = [p[0] for p in current_path]

        color = "seagreen" if snap["solved"] else "tomato"
        trail.set_data(cols, rows)
        trail.set_color(color)
        agent_dot.set_data([cols[-1]], [rows[-1]])
        agent_dot.set_color(color)
        agent_dot.set_markeredgecolor("darkgreen" if snap["solved"] else "darkred")

        status = "SOLVED" if snap["solved"] and step_idx == len(path) - 1 else ""
        if not snap["solved"] and step_idx == len(path) - 1:
            status = "FAILED"
        title.set_text(
            f"Episode {snap['episode']}  |  "
            f"Step: {step_idx}/{snap['steps']}  {status}"
        )
        return trail, agent_dot, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_frames),
        interval=interval, blit=True, repeat=False
    )

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved training progress GIF to {save_path}")

    plt.tight_layout()
    plt.show()
    plt.close(fig)