# RL Maze Solver

A comparison of three Reinforcement Learning algorithms **Q-Learning**, **DQN**, and **PPO** on a custom procedurally-generated maze environment. Agents learn to navigate from a start position to an exit while avoiding traps and minimizing steps.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Running the Agents](#running-the-agents)
  - [Train + Evaluate All Three](#train--evaluate-all-three-agents)
- [Interpreting Results](#interpreting-results)
- [Configuration](#configuration)

---

## Project Overview

Each maze is an 11×11 grid generated via randomized depth-first search (DFS). The environment supports:

- **Multiple routes** via random wall removal (30% of walls)
- **Traps** scattered across path cells (10% of cells)
- **Procedural seeds** for reproducible or randomized maze generation

Three RL agents are trained and evaluated in two modes:

| Mode | Description |
|------|-------------|
| **Single Maze** | Train and evaluate on a fixed maze (seed=42) |
| **Multi-Maze** | Train across 5 different mazes to test generalization |

---

## Project Structure

```
rl-maze-solver/
├── agents/
│   ├── q_learning.py       # Tabular Q-Learning with epsilon-greedy
│   ├── dqn_agent.py        # Double DQN with Dueling architecture
│   └── ppo_agent.py        # PPO with Actor-Critic and GAE
├── envs/
│   ├── maze_env.py         # Gymnasium-compatible maze environment
│   └── maze_generator.py   # DFS-based procedural maze generator
├── training/
│   ├── q_training.py       # Q-Learning training script
│   ├── dqn_training.py     # DQN training script
│   └── ppo_training.py     # PPO training script
├── utils/
│   └── visualization.py    # Maze rendering and GIF export
└── outputs/
    ├── q_learning/         # Q-Learning artifacts
    ├── dqn/                # DQN artifacts
    └── ppo/                # PPO artifacts
```

---

## Environment

### Maze Layout

| Cell Type | Description |
|-----------|-------------|
| White | Open path |
| Black | Wall (impassable) |
| Green | Agent start position |
| Red | Exit (goal) |
| Orange | Trap |
| Blue | Agent current position |

### Actions

| Action | Direction |
|--------|-----------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

### Reward Structure

| Event | Reward |
|-------|--------|
| Reaching the exit | +10 |
| Each step taken | -0.1 |
| Stepping into a trap | -1.0 (plus the -0.1 step penalty) |
| Hitting a wall (no movement) | -0.3 |

Episodes truncate after **500 steps**.

---

## Algorithms

### Q-Learning (Tabular)
- Classic lookup-table approach; state is `(row, col)` tuple
- Epsilon-greedy exploration with exponential decay
- Suitable baseline for small, discrete state spaces
- Key hyperparameters: `lr=0.1`, `gamma=0.99`, `epsilon_decay=0.995`

### DQN (Deep Q-Network)
- **Double DQN** with a **Dueling architecture** (separate Value and Advantage heads)
- **Experience Replay** buffer (500k capacity) with a target network
- State encoding: flattened 11×11 maze grid + normalized agent `(row, col)` = 123-dimensional input
- Key hyperparameters: `lr=5e-4`, `batch_size=256`, `target_update_freq=500`

### PPO (Proximal Policy Optimization)
- On-policy **Actor-Critic** with a shared network trunk
- **Generalized Advantage Estimation (GAE)** with `lambda=0.95`
- PPO-Clip objective with entropy regularization to encourage exploration
- Early stopping based on KL divergence and success-rate monitoring
- Key hyperparameters: `lr=2e-4`, `clip_ratio=0.2`, `n_steps=2048`, `ppo_epochs=10`

---

## Installation

**Prerequisites:** Python 3.8+

```bash
pip install numpy torch gymnasium matplotlib
```

Clone the repository and navigate to the project root:

```bash
git clone <repo-url>
cd rl-maze-solver
```

---

## Running the Agents

All scripts live in [training/](training/) and must be run from the **project root**.

Each script has two modes controlled by a single flag:

| Command | What happens |
|---------|-------------|
| `python training/<script>.py --train` | Train from scratch — runs single-maze then multi-maze automatically, saves all outputs |
| `python training/<script>.py` | Eval/test mode — loads the saved model and runs a test episode with a GIF |

---

### Train + Evaluate All Three Agents

Run these sequentially to reproduce the full experiment:

```bash
# 1. Q-Learning  (~10 000 episodes, fast)
python training/q_training.py --train

# 2. DQN  (~3 000 episodes)
python training/dqn_training.py --train

# 3. PPO  (~600 000 timesteps, slowest)
python training/ppo_training.py --train
```

Once training is done, run eval for each to generate fresh test GIFs and a printed summary:

```bash
python training/q_training.py
python training/dqn_training.py
python training/ppo_training.py
```

Eval mode prints three numbers to the console:

```
Evaluation (100 episodes):
  Avg Reward:   8.43
  Avg Steps:    34.2
  Success Rate: 91.0%
```

---

### Output Location

All artifacts are saved to `outputs/<algorithm>/`:

```
outputs/
├── q_learning/
│   ├── q_table_single.pkl
│   ├── q_table_multi.pkl
│   ├── single_maze_metrics.png
│   ├── multi_maze_metrics.png
│   ├── q_learning_single.gif
│   └── q_learning_multi.gif
├── dqn/
│   ├── dqn_single.pt
│   ├── dqn_multi.pt
│   ├── single_maze_metrics.png
│   ├── multi_maze_metrics.png
│   ├── dqn_single_test.gif
│   ├── dqn_multi_test.gif
│   └── config.json
└── ppo/
    ├── ppo_single.pt
    ├── ppo_multi.pt
    ├── single_maze_metrics.png
    ├── multi_maze_metrics.png
    ├── ppo_single_test.gif
    ├── ppo_multi_test.gif
    └── config.json
```

---

## Interpreting Results

### Metrics Plots (`*_metrics.png`)

Each algorithm saves a training metrics plot with multiple subplots. All plots show a **50-episode rolling average** to smooth noise.

#### Q-Learning and DQN — 4/6 panel plot:

| Subplot | What to Look For |
|---------|-----------------|
| **Episode Reward** | Should trend upward over time; a plateau near +10 means the agent consistently reaches the exit |
| **Steps per Episode** | Should decrease as the agent learns shorter paths; very low steps + high reward = optimal behavior |
| **Success Rate** | % of evaluation episodes where the agent reached the exit; above 80% is strong |
| **Epsilon** | Shows the exploration decay — as epsilon approaches 0.01 the agent is fully exploiting its learned policy |
| **Loss** (DQN only) | Training loss of the Q-network; expect an early spike then gradual reduction |

#### PPO — 6 panel plot:

| Subplot | What to Look For |
|---------|-----------------|
| **Episode Reward** | Upward trend indicates improving policy |
| **Steps per Episode** | Downward trend indicates more efficient navigation |
| **Success Rate** | Primary performance metric |
| **Actor Loss** | Measures how much the policy is being updated; large spikes may indicate instability |
| **Critic Loss** | Value function accuracy; should decrease as the critic learns better state estimates |
| **Entropy** | Policy randomness; if it collapses too early (below 0.1) the agent may stop exploring |

#### Multi-Maze Plots

Multi-maze plots include **vertical dashed lines** at maze transition boundaries (every 100 episodes for Q-Learning/DQN, every 60k timesteps for PPO). A drop in performance after a transition is expected — the agent is adapting to a new maze. Watch for recovery: a fast recovery indicates good generalization.

---

### GIF Visualizations (`*.gif`)

The GIFs show the trained agent navigating the maze in real time (150ms per frame).

- **Blue cell** = agent's current position
- **Green cell** = start
- **Red cell** = exit (goal)
- **Orange cells** = traps to avoid

A successful run ends with the blue cell reaching red. Look for smooth, direct paths without unnecessary backtracking — this indicates a well-trained policy.

---

### Saved Models

| File | Format | Contains |
|------|--------|----------|
| `q_table_*.pkl` | Python pickle | Q-value dictionary keyed by `(row, col)` state |
| `dqn_*.pt` | PyTorch checkpoint | `online_net`, `target_net`, `optimizer`, `epsilon`, `learn_steps` |
| `ppo_*.pt` | PyTorch checkpoint | Actor-Critic `state_dict`, `optimizer` state |

---

## Configuration

Each algorithm's hyperparameters are stored in `outputs/<algorithm>/config.json` and are loaded at training time. Key parameters you may want to adjust:

| Parameter | Location | Effect |
|-----------|----------|--------|
| `width` / `height` | `env` section | Maze size (must be odd) |
| `trap_fraction` | `env` section | Density of traps (0.0–1.0) |
| `wall_removal_fraction` | `env` section | How many walls to remove for alternate paths |
| `max_steps` | `env` section | Episode truncation limit |
| `episodes` / `total_timesteps` | `training` section | Total training budget |
| `multi_maze_seeds` | `training` section | Seeds used for multi-maze training |
| `learning_rate` / `lr` | `agent` section | Step size for gradient/table updates |
| `epsilon_decay` | `agent` section | Exploration decay rate (Q-Learning / DQN) |
| `entropy_coef` | `agent` section | Entropy bonus weight for PPO exploration |
