import numpy as np


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, chosen_arm):
        self.regret += self.bandit.best_prob - self.bandit.probs[chosen_arm]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def run(self, n_steps):
        for _ in range(n_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class RandomSolver(Solver):
    def run_one_step(self):
        return np.random.randint(0, self.bandit.K)


class EpsilonGreedySolver(Solver):
    def __init__(self, bandit, epsilon: float = 0.01, init_prob: float = 1.0):
        super().__init__(bandit)
        self.epsilon = epsilon
        self.q_values = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.q_values)
        r = self.bandit.step(k)
        self.update_q_value(k, r)
        return k

    def update_q_value(self, chosen_arm, reward):
        n = self.counts[chosen_arm] + 1  # +1 because counts is updated after this call
        self.q_values[chosen_arm] += 1.0 / n * (reward - self.q_values[chosen_arm])


class DecayEpsilonGreedySolver(Solver):
    def __init__(self, bandit, init_prob: float = 1.0):
        super().__init__(bandit=bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class UCBSolver(Solver):
    def __init__(self, bandit, init_prob: float = 1.0):
        super().__init__(bandit=bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        ucb_values = self.estimates + np.sqrt(2 * np.log(self.total_count) / (self.counts + 1e-5))
        k = np.argmax(ucb_values)
        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


class ThompsonSamplingSolver(Solver):
    def __init__(self, bandit):
        super().__init__(bandit=bandit)
        self.successes = np.zeros(self.bandit.K)
        self.failures = np.zeros(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        k = np.argmax(samples)
        r = self.bandit.step(k)
        if r == 1.0:
            self.successes[k] += 1
        else:
            self.failures[k] += 1
        return k
