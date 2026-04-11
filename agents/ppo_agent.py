"""
Proximal Policy Optimization (PPO) Agent for MazeRL Project

On-policy Actor-Critic with GAE, PPO-Clip objective, and entropy bonus.
State encoding mirrors DQNAgent: flattened maze grid (normalised) + agent position.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class RolloutBuffer:
    """Fixed-horizon buffer for on-policy PPO trajectories.

    Collects exactly n_steps transitions (potentially spanning multiple episodes).
    Call finalise(last_value, gamma, gae_lambda) once full to compute GAE
    advantages and discounted returns, then iterate over mini-batches with
    get_batches(batch_size). Call clear() before the next rollout.

    Args:
        n_steps: Rollout horizon — transitions collected before each PPO update.
        state_size: Length of the flat state vector.
    """

    def __init__(self, n_steps, state_size):
        self.n_steps    = n_steps
        self._pos       = 0

        self.states    = np.zeros((n_steps, state_size), dtype=np.float32)
        self.actions   = np.zeros(n_steps,               dtype=np.int64)
        self.rewards   = np.zeros(n_steps,               dtype=np.float32)
        self.dones     = np.zeros(n_steps,               dtype=np.float32)
        self.log_probs = np.zeros(n_steps,               dtype=np.float32)
        self.values    = np.zeros(n_steps,               dtype=np.float32)

        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

    def push(self, state, action, reward, done, log_prob, value):
        p = self._pos
        self.states[p]    = state
        self.actions[p]   = action
        self.rewards[p]   = reward
        self.dones[p]     = done
        self.log_probs[p] = log_prob
        self.values[p]    = value
        self._pos        += 1

    def is_full(self):
        return self._pos >= self.n_steps

    def finalise(self, last_value, gamma, gae_lambda):
        """Compute GAE advantages and bootstrapped returns.

        next_values[t] = V(s_{t+1}):
            - For t < n_steps-1: self.values[t+1]
            - For t = n_steps-1: last_value (bootstrap if episode is ongoing,
              or 0.0 if the last transition was terminal/truncated)

        GAE: A_t = δ_t + (γλ)(1 - done_t) * A_{t+1}
             δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)

        When done_t=1 (episode boundary), future value and advantage are zeroed
        so advantages don't propagate across episodes.

        Args:
            last_value: V(s) for the state after the final stored transition.
                        Pass 0.0 if that transition was terminal or truncated.
            gamma: Discount factor.
            gae_lambda: GAE smoothing (0 → TD(0), 1 → Monte-Carlo).
        """
        next_values      = np.empty(self.n_steps, dtype=np.float32)
        next_values[:-1] = self.values[1:]
        next_values[-1]  = last_value

        gae = 0.0
        for t in reversed(range(self.n_steps)):
            non_terminal       = 1.0 - self.dones[t]
            delta              = self.rewards[t] + gamma * next_values[t] * non_terminal - self.values[t]
            gae                = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yield randomly shuffled mini-batches.

        Yields:
            Tuple of numpy arrays: (states, actions, old_log_probs, advantages, returns).
        """
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                self.states[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.advantages[idx],
                self.returns[idx],
            )

    def clear(self):
        self._pos = 0


class ActorCriticNetwork(nn.Module):
    """Shared-trunk Actor-Critic network for PPO.

    Architecture
    ------------
    A shared trunk extracts features, then splits into:

        Actor head  — logits over discrete actions (Categorical policy π(a|s))
        Critic head — scalar state value V(s)

    Sharing the trunk lets both policy and value function benefit from
    the same learned features, improving sample efficiency.

    LayerNorm + ELU keeps training stable as value targets shift during
    on-policy updates.  Orthogonal initialisation is standard for PPO.

    Args:
        state_size: Length of the flat input vector.
        action_size: Number of discrete actions.
        hidden_sizes: Tuple of hidden layer widths for the shared trunk.
                      Any depth is supported, e.g. (256, 128) or (512, 256, 128).
                      The last value is the width fed into both heads.
    """

    def __init__(self, state_size, action_size, hidden_sizes=(256, 128)):
        super().__init__()
        head_w = hidden_sizes[-1]

        trunk_layers = []
        in_dim = state_size
        for h in hidden_sizes:
            trunk_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        # Small init on actor head keeps early actions near-uniform
        self.actor_head = nn.Linear(head_w, action_size)

        self.critic_head = nn.Sequential(
            nn.Linear(head_w, head_w // 2),
            nn.Tanh(),
            nn.Linear(head_w // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        for m in self.critic_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Returns (logits, value)."""
        features = self.trunk(x)
        return self.actor_head(features), self.critic_head(features).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        """Sample or evaluate an action.

        Args:
            x: State tensor of shape (batch, state_size).
            action: If provided, evaluate log_prob/entropy for this action
                    (used during the PPO update). If None, sample from π.

        Returns:
            (action, log_prob, entropy, value)
        """
        logits, value = self.forward(x)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x):
        _, value = self.forward(x)
        return value


class PPOAgent:
    """On-policy PPO agent for discrete-action maze environments.

    State encoding is identical to DQNAgent: flattened normalised maze grid
    concatenated with the normalised agent position (row/H, col/W).

    Exposes get_best_action(obs_tuple) for compatibility with existing
    visualisation utilities.

    Training loop (caller's responsibility):
        1. Call choose_action_vec() each step to get (action, log_prob, value).
        2. Call store() to push the transition into the rollout buffer.
        3. When buffer_full() returns True, call update(last_value) to run
           PPO-Clip gradient updates and clear the buffer.

    Args:
        maze_grid: numpy ndarray (H, W) of cell values for the current maze.
        action_size: Number of discrete actions (default 4).
        lr: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE smoothing parameter (0 → TD(0), 1 → Monte-Carlo).
        clip_ratio: PPO clipping epsilon.
        value_coef: Weight for the critic loss term.
        entropy_coef: Weight for the entropy bonus (encourages exploration).
        max_grad_norm: Gradient clipping norm.
        n_steps: Rollout horizon — transitions collected before each PPO update.
        ppo_epochs: Gradient passes over each rollout.
        mini_batch_size: Mini-batch size for PPO updates.
        hidden_sizes: (trunk_width, head_width) for ActorCriticNetwork.
        device: 'cpu', 'cuda', or None (auto-detect).
    """

    def __init__(
        self,
        maze_grid,
        action_size=4,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_steps=2048,
        ppo_epochs=4,
        mini_batch_size=64,
        hidden_sizes=(256, 128),
        target_kl=None,
        device=None,
    ):
        self.action_size     = action_size
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_ratio      = clip_ratio
        self.value_coef      = value_coef
        self.entropy_coef    = entropy_coef
        self.max_grad_norm   = max_grad_norm
        self.n_steps         = n_steps
        self.ppo_epochs      = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl       = target_kl  # stop epochs early if approx KL exceeds this

        self.state_size = maze_grid.size + 2
        self._update_maze_cache(maze_grid)

        self.device = (
            torch.device(device) if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.net       = ActorCriticNetwork(self.state_size, action_size, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer(n_steps, self.state_size)

    # --- State encoding (identical to DQNAgent) ----------------------------

    def _update_maze_cache(self, maze_grid):
        self._h = maze_grid.shape[0]
        self._w = maze_grid.shape[1]
        self._maze_flat  = (maze_grid.astype(np.float32) / 4.0).flatten()
        self._encode_buf = np.empty(self.state_size, dtype=np.float32)
        self._encode_buf[:-2] = self._maze_flat

    def _encode(self, obs_tuple):
        out = self._encode_buf.copy()
        row, col = obs_tuple
        out[-2] = row / self._h
        out[-1] = col / self._w
        return out

    def set_maze(self, maze_grid):
        """Update stored maze when switching to a different maze.

        Args:
            maze_grid: numpy ndarray (H, W) of the new maze.
        """
        self._update_maze_cache(maze_grid)

    # --- Action selection ---------------------------------------------------

    def choose_action(self, obs_tuple):
        """Sample an action from the current stochastic policy.

        Args:
            obs_tuple: Agent position (row, col).

        Returns:
            (action, log_prob, value) as Python scalars.
        """
        return self.choose_action_vec(self._encode(obs_tuple))

    def choose_action_vec(self, state_vec):
        """Sample action from a pre-encoded state vector.

        Avoids redundant encoding when the caller already holds state_vec.

        Args:
            state_vec: Pre-encoded float32 numpy array of length state_size.

        Returns:
            (action, log_prob, value) as Python scalars.
        """
        t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action_and_value(t)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def get_best_action(self, obs_tuple):
        """Return the greedy (argmax) action — no sampling (for visualisation).

        Args:
            obs_tuple: Agent position (row, col).
        """
        state_vec = self._encode(obs_tuple)
        t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(t)
        return int(logits.argmax(dim=1).item())

    def get_value(self, state_vec):
        """Estimate V(s) for a pre-encoded state vector (used for bootstrapping).

        Args:
            state_vec: Pre-encoded float32 numpy array.
        """
        t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.net.get_value(t).item())

    # --- Buffer interaction -------------------------------------------------

    def store(self, state_vec, action, reward, done, log_prob, value):
        """Push one transition into the rollout buffer.

        Use done = terminated OR truncated so GAE does not propagate
        across episode boundaries.

        Args:
            state_vec: Pre-encoded state vector for the current step.
            action: Action taken.
            reward: Reward received.
            done: True if the episode ended (terminal or truncated).
            log_prob: Log-probability of the action under the current policy.
            value: V(s) estimate for the current state.
        """
        self.buffer.push(state_vec, action, reward, float(done), log_prob, value)

    def buffer_full(self):
        return self.buffer.is_full()

    # --- PPO update ---------------------------------------------------------

    def update(self, last_value):
        """Finalise the rollout and run PPO-Clip gradient updates.

        Steps:
            1. Compute GAE advantages and discounted returns.
            2. Normalise advantages over the full rollout.
            3. Run ppo_epochs passes of mini-batch clipped-policy updates.
            4. Clear the buffer.

        Args:
            last_value: V(s) bootstrapped for the state after the last stored
                        transition. Pass 0.0 if the last step was terminal/truncated.

        Returns:
            Dictionary with scalar metrics:
                'actor_loss', 'critic_loss', 'entropy', 'total_loss'.
        """
        self.buffer.finalise(last_value, self.gamma, self.gae_lambda)

        # Normalise advantages for stable gradient magnitudes
        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_losses  = []
        critic_losses = []
        entropies     = []

        for _ in range(self.ppo_epochs):
            epoch_kls = []
            for batch in self.buffer.get_batches(self.mini_batch_size):
                s_b, a_b, olp_b, adv_b, ret_b = batch

                s   = torch.from_numpy(s_b).to(self.device)
                a   = torch.from_numpy(a_b).to(self.device)
                olp = torch.from_numpy(olp_b).to(self.device)
                adv = torch.from_numpy(adv_b).to(self.device)
                ret = torch.from_numpy(ret_b).to(self.device)

                _, log_prob, entropy, value = self.net.get_action_and_value(s, a)

                # PPO-Clip actor loss
                ratio      = torch.exp(log_prob - olp)
                clipped    = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                actor_loss = -torch.min(ratio * adv, clipped * adv).mean()

                # MSE critic loss — Huber with default delta=1.0 is too
                # aggressive for maze returns that span [-50, +10].
                critic_loss = nn.functional.mse_loss(value, ret)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropies.append(entropy.mean().item())

                # Track approx KL for early epoch stopping
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = 0.5 * ((log_prob - olp) ** 2).mean().item()
                    epoch_kls.append(approx_kl)

            # Stop epochs early if policy is drifting too far from the rollout policy
            if self.target_kl is not None and float(np.mean(epoch_kls)) > self.target_kl:
                break

        self.buffer.clear()

        return {
            "actor_loss":  float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy":     float(np.mean(entropies)),
            "total_loss":  float(np.mean(actor_losses)) + self.value_coef * float(np.mean(critic_losses)),
        }

    # --- Persistence --------------------------------------------------------

    def save(self, filepath):
        """Save network weights and optimizer state to a .pt file.

        Args:
            filepath: Destination path (parent directories created automatically).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "net":       self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )
        print(f"PPO model saved to {filepath}")

    def load(self, filepath):
        """Load network weights and optimizer state from a .pt file.

        Args:
            filepath: Path to the saved .pt checkpoint.
        """
        ckpt = torch.load(filepath, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
