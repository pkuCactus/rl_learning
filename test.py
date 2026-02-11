import numpy as np
from matplotlib import pyplot as plt
from basic.bandit import BernoulliBandit
from basic.rl_solver import DecayEpsilonGreedySolver, RandomSolver, EpsilonGreedySolver


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('%d-armed bandit problem' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    np.random.seed(1)
    bandit_10_armed = BernoulliBandit(K=10)
    print("Best arm: %d, Best probability: %.4f" % (bandit_10_armed.best_arm, bandit_10_armed.best_prob))
    np.random.seed(1)
    random_solver = RandomSolver(bandit_10_armed)
    random_solver.run(n_steps=5000)
    print("Random Solver cumulative regret: %.4f" % random_solver.regret)
    greedy_solvers = [EpsilonGreedySolver(bandit_10_armed, epsilon=epsilon) for epsilon in [1e-4, 1e-3, 1e-2]]
    for greedy_solver in greedy_solvers:
        np.random.seed(1)
        greedy_solver.run(n_steps=5000)
        print("Epsilon-Greedy Solver (epsilon=%.4f) cumulative regret: %.4f" % (greedy_solver.epsilon, greedy_solver.regret))
    decay_greedy_solver = DecayEpsilonGreedySolver(bandit_10_armed)
    np.random.seed(1)
    decay_greedy_solver.run(n_steps=5000)
    print("Decay Epsilon-Greedy Solver cumulative regret: %.4f" % decay_greedy_solver.regret)
    plot_results(greedy_solvers + [decay_greedy_solver], [f'Epsilon-Greedy Solver (epsilon={epsilon})' for epsilon in [1e-4, 1e-3, 1e-2]] + ['Decay Epsilon-Greedy Solver'])