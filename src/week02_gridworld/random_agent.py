import numpy as np


class RandomAgent:
    def __init__(self, num_actions=4, seed=42):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def select_action(self, state):
        return int(self.rng.integers(0, self.num_actions))