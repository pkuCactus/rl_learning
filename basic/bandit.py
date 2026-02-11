import numpy as np

class BernoulliBandit:
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        self.best_arm = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_arm]
        self.K = K

    def step(self, chosen_arm):
        if np.random.rand() < self.probs[chosen_arm]:
            return 1.0
        else:
            return 0.0
