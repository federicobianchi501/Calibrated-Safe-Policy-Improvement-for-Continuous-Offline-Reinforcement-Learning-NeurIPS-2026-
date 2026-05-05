# =============================================================================
# Anonymous authors. Affiliations and acknowledgements will be added on
# acceptance of the associated paper. Submitted to NeurIPS 2026 (double-blind).
# =============================================================================
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            actions=self.act_buf[idxs],
            rewards=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            dones=self.done_buf[idxs],
        )
