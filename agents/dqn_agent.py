"""
Deep Q-Network (DQN) Agent for MazeRL Project

Double DQN with Dueling architecture, experience replay, and a target network.
State: flattened maze grid (normalised) + normalised agent position.
Scales to larger and higher-dimensional state spaces versus tabular Q-Learning.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """Pre-allocated numpy circular buffer for (s, a, r, s', done) transitions.

    Uses fixed numpy arrays instead of a deque so both push (O(1) index write)
    and sample (vectorised numpy fancy-index) are fast regardless of buffer size.
    A deque requires O(n) random access which makes sampling progressively slower
    as the buffer grows.

    Args:
        capacity: Maximum number of transitions to store.
        state_size: Length of each flat state vector.
    """

    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self._pos  = 0
        self._size = 0

        self.states      = np.zeros((capacity, state_size), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        p = self._pos
        self.states[p]      = state
        self.next_states[p] = next_state
        self.actions[p]     = action
        self.rewards[p]     = reward
        self.dones[p]       = done
        self._pos  = (p + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a random mini-batch via vectorised numpy indexing.

        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones).
        """
        idx = np.random.randint(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self._size


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network with LayerNorm and ELU activations.

    Architecture
    ------------
    A shared trunk extracts features, then splits into two streams:

        Value stream     V(s)         
        Advantage stream A(s, a)       

    Combined as: Q(s, a) = V(s) + A(s, a) - mean_a(A(s, a))

    This separation makes value estimation more stable because the network
    can learn state values independently of per-action advantages.

    LayerNorm stabilises training when Q-value targets shift during learning.
    ELU avoids dying neurons (unlike ReLU) and has smoother gradients.

    Args:
        state_size: Length of the flat input vector.
        action_size: Number of discrete actions.
        hidden_sizes: Two-element tuple (trunk_width, head_width).
    """

    def __init__(self, state_size, action_size, hidden_sizes=(256, 128)):
        super().__init__()
        trunk_w, head_w = hidden_sizes

        self.trunk = nn.Sequential(
            nn.Linear(state_size, trunk_w),
            nn.LayerNorm(trunk_w),
            nn.ELU(),
            nn.Linear(trunk_w, head_w),
            nn.LayerNorm(head_w),
            nn.ELU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(head_w, head_w // 2),
            nn.ELU(),
            nn.Linear(head_w // 2, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(head_w, head_w // 2),
            nn.ELU(),
            nn.Linear(head_w // 2, action_size),
        )

    def forward(self, x):
        features  = self.trunk(x)
        value     = self.value_head(features)
        advantage = self.advantage_head(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    """
    Double DQN agent with Dueling architecture for discrete-action maze environments.

    The state is encoded as the flattened maze grid (cell values / 4.0) concatenated
    with the normalised agent position (row / H, col / W). This gives the network
    full knowledge of maze structure and scales to any maze size without a Q-table.

    The maze flat vector is cached on set_maze() so _encode() costs only one
    array concatenation per step.

    The agent exposes get_best_action(obs_tuple) with the same signature as
    QLearningAgent so it works with existing visualisation utilities.

    Args:
        maze_grid: numpy ndarray (H, W) of cell values for the current maze.
        action_size: Number of discrete actions (default 4).
        lr: Adam learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Multiplicative decay applied per episode.
        batch_size: Mini-batch size drawn from the replay buffer.
        buffer_capacity: Maximum replay buffer size.
        min_replay_size: Minimum buffer occupancy before learning begins.
        target_update_freq: Learn-steps between hard target-network syncs.
        hidden_sizes: (trunk_width, head_width) for DuelingQNetwork.
    """

    def __init__(
        self,
        maze_grid,
        action_size=4,
        lr=5e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        batch_size=256,
        buffer_capacity=500_000,
        min_replay_size=2_000,
        target_update_freq=500,
        hidden_sizes=(256, 128),
        learn_every=4,
        device=None,
    ):
        self.action_size     = action_size
        self.gamma           = gamma
        self.epsilon         = epsilon
        self.epsilon_min     = epsilon_min
        self.epsilon_decay   = epsilon_decay
        self.batch_size      = batch_size
        self.min_replay_size = min_replay_size
        self.target_update_freq = target_update_freq
        self.learn_every     = learn_every
        self._learn_steps    = 0

        self.state_size = maze_grid.size + 2
        self._update_maze_cache(maze_grid)
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._use_amp = self.device.type == "cuda"

        self.online_net = DuelingQNetwork(self.state_size, action_size, hidden_sizes).to(self.device)
        self.target_net = DuelingQNetwork(self.state_size, action_size, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()
        self.scaler    = torch.amp.GradScaler(self.device.type, enabled=self._use_amp)
        self.buffer    = ReplayBuffer(buffer_capacity, self.state_size)

    def _update_maze_cache(self, maze_grid):
        """Precompute the normalised flat maze vector and pre-fill the encode buffer."""
        self._h = maze_grid.shape[0]
        self._w = maze_grid.shape[1]
        self._maze_flat  = (maze_grid.astype(np.float32) / 4.0).flatten()

        self._encode_buf = np.empty(self.state_size, dtype=np.float32)
        self._encode_buf[:-2] = self._maze_flat

    def _encode(self, obs_tuple):
        """Encode a (row, col) observation into a flat float32 state vector.

        Uses a pre-filled buffer so only a fast memcpy + two scalar writes
        are needed per call instead of copying 441 floats from scratch.
        """
        out = self._encode_buf.copy()   
        row, col = obs_tuple
        out[-2] = row / self._h
        out[-1] = col / self._w
        return out

    def set_maze(self, maze_grid):
        """Update the stored maze when switching to a different maze.

        Args:
            maze_grid: numpy ndarray (H, W) of the new maze.
        """
        self._update_maze_cache(maze_grid)

    def choose_action(self, obs_tuple):
        """Select an action using epsilon-greedy policy.

        Args:
            obs_tuple: Current agent position as (row, col).
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return self._greedy_action(self._encode(obs_tuple))

    def choose_action_vec(self, state_vec):
        """Epsilon-greedy using a pre-encoded state vector (avoids redundant encoding).

        Use this in the training loop together with a single _encode() call per step
        so the same vector is reused for both action selection and buffer storage.

        Args:
            state_vec: Pre-encoded float32 numpy array of length state_size.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return self._greedy_action(state_vec)

    def get_best_action(self, obs_tuple):
        """Return the greedy action (no exploration) for a given position.

        Args:
            obs_tuple: Agent position as (row, col).
        """
        return self._greedy_action(self._encode(obs_tuple))

    def _greedy_action(self, state_vec):
        t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.online_net(t).argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        """Encode a transition and push it into the replay buffer.

        Args:
            obs: Current position (row, col).
            action: Action taken.
            reward: Reward received.
            next_obs: Next position (row, col).
            done: Whether the episode ended.
        """
        self.buffer.push(
            self._encode(obs),
            action,
            reward,
            self._encode(next_obs),
            float(done),
        )

    def learn(self):
        """Sample a mini-batch and perform one Double-DQN gradient update.

        Waits until the replay buffer has at least min_replay_size transitions
        so early Q-value estimates are based on reasonably diverse experience.

        Uses AMP (automatic mixed precision) on CUDA for faster computation.

        Double DQN: online network selects the next action, target network
        evaluates it — reducing Q-value overestimation bias.

        Returns:
            Loss value as a float, or None if the buffer is not yet ready.
        """
        if len(self.buffer) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        s  = torch.from_numpy(states).to(self.device,  non_blocking=True)
        a  = torch.from_numpy(actions).to(self.device,  non_blocking=True)
        r  = torch.from_numpy(rewards).to(self.device,  non_blocking=True)
        ns = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        d  = torch.from_numpy(dones).to(self.device,   non_blocking=True)

        with torch.amp.autocast(self.device.type, enabled=self._use_amp):
            q_pred = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                best_next_a = self.online_net(ns).argmax(dim=1, keepdim=True)
                q_next      = self.target_net(ns).gather(1, best_next_a).squeeze(1)
                q_target    = r + self.gamma * q_next * (1.0 - d)

            loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self._learn_steps += 1
        if self._learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save model weights and training state to a .pt file.

        Args:
            filepath: Destination path (parent directories are created automatically).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "scaler":      self.scaler.state_dict(),
                "epsilon":     self.epsilon,
                "learn_steps": self._learn_steps,
            },
            filepath,
        )
        print(f"DQN model saved to {filepath}")

    def load(self, filepath):
        """Load model weights and training state from a .pt file.

        Args:
            filepath: Path to the saved .pt checkpoint.
        """
        ckpt = torch.load(filepath, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.epsilon      = ckpt["epsilon"]
        self._learn_steps = ckpt["learn_steps"]
