"""
Custom Gymnasium Environment for MazeRL Project

Actions: 0=Up, 1=Down, 2=Left, 3=Right
Observation: Agent's (row, col) position
Rewards: +10 exit, -0.1 per step, -1 trap, -0.3 wall bump (no movement)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from envs.maze_generator import MazeGenerator


class MazeEnv(gym.Env):
    """
    Grid-based maze environment compatible with Gymnasium API.

    Args:
        width, height: Maze dimensions (must be odd, auto-adjusted if even).
        trap_fraction: Fraction of path cells turned into passable traps.
        wall_removal_fraction: Fraction of interior walls removed for multiple routes.
        max_steps: Maximum steps before episode is truncated.
        seed: Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    REWARD_EXIT = 10.0
    REWARD_STEP = -0.1
    REWARD_TRAP = -1.0
    REWARD_WALL = -0.3

    def __init__(self, width=11, height=11, trap_fraction=0.10,
                 wall_removal_fraction=0.3, max_steps=500, seed=None,
                 render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        self.generator = MazeGenerator(
            width=width, height=height,
            trap_fraction=trap_fraction,
            wall_removal_fraction=wall_removal_fraction,
            seed=seed
        )

        self.grid, self.start_pos, self.exit_pos = self.generator.generate()
        self.height, self.width = self.grid.shape

        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.height - 1, self.width - 1]),
            dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)

        self.agent_pos = None
        self.steps_taken = 0
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        """
        Reset environment with a new or existing maze.

        Args:
            seed: Optional random seed.
            options: Pass {"new_maze": True} to generate a fresh maze.
        """
        super().reset(seed=seed)

        if options and options.get("new_maze", False):
            self.grid, self.start_pos, self.exit_pos = self.generator.generate()

        self.agent_pos = list(self.start_pos)
        self.steps_taken = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one action in the maze.

        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right.
        """
        self.steps_taken += 1
        dr, dc = self.ACTION_DELTAS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc

        cell = self.grid[new_row][new_col]

        if cell == MazeGenerator.WALL:
            reward = self.REWARD_WALL
        else:
            self.agent_pos = [new_row, new_col]
            if cell == MazeGenerator.EXIT:
                reward = self.REWARD_EXIT
            elif cell == MazeGenerator.TRAP:
                reward = self.REWARD_TRAP + self.REWARD_STEP
            else:
                reward = self.REWARD_STEP

        terminated = (self.grid[self.agent_pos[0]][self.agent_pos[1]] == MazeGenerator.EXIT)
        truncated = (self.steps_taken >= self.max_steps)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        """Return agent's current position as observation."""
        return np.array(self.agent_pos, dtype=np.int32)

    def _get_info(self):
        """Return auxiliary info about the current state."""
        return {
            "steps": self.steps_taken,
            "agent_pos": tuple(self.agent_pos),
            "distance_to_exit": abs(self.agent_pos[0] - self.exit_pos[0])
                              + abs(self.agent_pos[1] - self.exit_pos[1])
        }

    def render(self):
        """
        Render the maze with the agent's current position.

        Returns:
            RGB array if render_mode is "rgb_array", otherwise displays plot.
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self.ax.clear()

        cmap = mcolors.ListedColormap(["white", "black", "green", "red", "orange"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        self.ax.imshow(self.grid, cmap=cmap, norm=norm, interpolation="nearest")

        self.ax.plot(
            self.agent_pos[1], self.agent_pos[0],
            "bo", markersize=12, markeredgecolor="darkblue", markeredgewidth=2
        )

        self.ax.set_title(f"Step: {self.steps_taken}", fontsize=14, fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for x in range(self.width + 1):
            self.ax.axvline(x - 0.5, color="gray", linewidth=0.3)
        for y in range(self.height + 1):
            self.ax.axhline(y - 0.5, color="gray", linewidth=0.3)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

        plt.pause(0.05)

    def close(self):
        """Clean up matplotlib resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None