"""
Maze Generator for MazeRL Project
Uses DFS (Recursive Backtracker) with wall removal to create multi-path mazes.

Cell Values: 0=Path, 1=Wall, 2=Start, 3=Exit, 4=Trap
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class MazeGenerator:
    """
    Procedural maze generator using DFS with wall removal for open layouts.

    Args:
        width, height: Maze dimensions (must be odd, auto-adjusted if even).
        trap_fraction: Fraction of path cells turned into passable traps (0.0-1.0).
        wall_removal_fraction: Fraction of interior walls removed to create loops (0.0-1.0).
        seed: Random seed for reproducibility.
    """

    WALL = 1
    PATH = 0
    START = 2
    EXIT = 3
    TRAP = 4

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    DFS_DIRECTIONS = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    def __init__(self, width=11, height=11, trap_fraction=0.10,
                 wall_removal_fraction=0.3, seed=None):
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        self.trap_fraction = np.clip(trap_fraction, 0.0, 1.0)
        self.wall_removal_fraction = np.clip(wall_removal_fraction, 0.0, 1.0)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.grid = None
        self.start_pos = None
        self.exit_pos = None

    def generate(self):
        """Generate a new maze. Returns (grid, start_pos, exit_pos)."""
        self.grid = np.ones((self.height, self.width), dtype=np.int32)
        self._carve_passages()
        self._remove_extra_walls()
        self._place_start_and_exit()
        self._place_traps()
        return self.grid.copy(), self.start_pos, self.exit_pos

    def _carve_passages(self):
        """Carve passages using iterative DFS."""
        start_row, start_col = 1, 1
        self.grid[start_row][start_col] = self.PATH
        stack = [(start_row, start_col)]

        while stack:
            current_row, current_col = stack[-1]

            neighbors = []
            for dr, dc in self.DFS_DIRECTIONS:
                nr, nc = current_row + dr, current_col + dc
                if 0 < nr < self.height - 1 and 0 < nc < self.width - 1:
                    if self.grid[nr][nc] == self.WALL:
                        neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = random.choice(neighbors)
                self.grid[current_row + dr // 2][current_col + dc // 2] = self.PATH
                self.grid[nr][nc] = self.PATH
                stack.append((nr, nc))
            else:
                stack.pop()

    def _remove_extra_walls(self):
        """Remove interior walls between path cells to create loops and alternate routes."""
        if self.wall_removal_fraction <= 0:
            return

        interior_walls = []
        for r in range(1, self.height - 1):
            for c in range(1, self.width - 1):
                if self.grid[r][c] == self.WALL:
                    adjacent_paths = sum(
                        1 for dr, dc in self.DIRECTIONS
                        if 0 <= r + dr < self.height and 0 <= c + dc < self.width
                        and self.grid[r + dr][c + dc] == self.PATH
                    )
                    if adjacent_paths >= 2:
                        interior_walls.append((r, c))

        num_to_remove = int(len(interior_walls) * self.wall_removal_fraction)
        if num_to_remove == 0:
            return

        for r, c in random.sample(interior_walls, min(num_to_remove, len(interior_walls))):
            self.grid[r][c] = self.PATH

    def _place_start_and_exit(self):
        """Place start at (1,1) and exit at the farthest open cell."""
        self.start_pos = (1, 1)
        self.grid[self.start_pos[0]][self.start_pos[1]] = self.START

        best_pos = None
        best_dist = -1
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == self.PATH and (r + c) > best_dist:
                    best_dist = r + c
                    best_pos = (r, c)

        self.exit_pos = best_pos if best_pos else (self.height - 2, self.width - 2)
        self.grid[self.exit_pos[0]][self.exit_pos[1]] = self.EXIT

    def _place_traps(self):
        """Scatter passable traps on random path cells (excluding start and exit)."""
        if self.trap_fraction <= 0:
            return

        path_cells = list(zip(*np.where(self.grid == self.PATH)))
        num_traps = int(len(path_cells) * self.trap_fraction)

        if num_traps == 0 or len(path_cells) == 0:
            return

        for r, c in random.sample(path_cells, min(num_traps, len(path_cells))):
            self.grid[r][c] = self.TRAP

    def render(self, ax=None, title="Generated Maze", show=True):
        """
        Visualize the maze grid using Matplotlib.

        Args:
            ax: Matplotlib Axes to draw on. Creates a new figure if None.
            title: Plot title.
            show: Whether to call plt.show().
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        cmap = mcolors.ListedColormap(["white", "black", "green", "red", "orange"])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(self.grid, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        for x in range(self.width + 1):
            ax.axvline(x - 0.5, color="gray", linewidth=0.3)
        for y in range(self.height + 1):
            ax.axhline(y - 0.5, color="gray", linewidth=0.3)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="white", edgecolor="black", label="Path"),
            Patch(facecolor="black", edgecolor="black", label="Wall"),
            Patch(facecolor="green", edgecolor="black", label="Start"),
            Patch(facecolor="red", edgecolor="black", label="Exit"),
            Patch(facecolor="orange", edgecolor="black", label="Trap (passable)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

        if show:
            plt.tight_layout()
            plt.show()

        return ax


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    configs = [
        {"width": 11, "height": 11, "trap_fraction": 0.10, "wall_removal_fraction": 0.3, "seed": 42},
        {"width": 15, "height": 15, "trap_fraction": 0.15, "wall_removal_fraction": 0.4, "seed": 99},
        {"width": 21, "height": 21, "trap_fraction": 0.10, "wall_removal_fraction": 0.35, "seed": 7},
    ]

    for i, cfg in enumerate(configs):
        gen = MazeGenerator(**cfg)
        grid, start, exit_pos = gen.generate()
        gen.render(
            ax=axes[i],
            title=(f"Maze {cfg['width']}x{cfg['height']}  |  "
                   f"traps={cfg['trap_fraction']*100:.0f}%  "
                   f"openness={cfg['wall_removal_fraction']*100:.0f}%"),
            show=False,
        )

        total_walkable = np.sum(np.isin(grid, [MazeGenerator.PATH, MazeGenerator.START,
                                                MazeGenerator.EXIT, MazeGenerator.TRAP]))
        total_wall = np.sum(grid == MazeGenerator.WALL)
        total_trap = np.sum(grid == MazeGenerator.TRAP)
        print(f"Maze {i+1}: size={cfg['width']}x{cfg['height']}, "
              f"start={start}, exit={exit_pos}, "
              f"walkable={total_walkable}, walls={total_wall}, traps={total_trap}")

    plt.tight_layout()
    plt.savefig("maze_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved preview to maze_samples.png")