# =============================================================================
# Calibrated Safe Policy Improvement for Continuous Offline Reinforcement Learning
# Federico Bianchi, Alessandro Farinelli, Alberto Castellini
# NeurIPS 2026
# =============================================================================
"""Fixed-size FIFO replay buffer for flat observations."""
import numpy as np

class ReplayBuffer:
    """Ring buffer of (obs, action, reward, next_obs, done) transitions stored as float32 arrays."""

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        """Store one transition, overwriting the oldest entry when full."""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Sample `batch_size` transitions uniformly with replacement.

        Returns a dict with keys obs, actions, rewards, next_obs, dones.
        """
        idxs = np.random.randint(0, self.size, batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            actions=self.act_buf[idxs],
            rewards=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            dones=self.done_buf[idxs],
        )
