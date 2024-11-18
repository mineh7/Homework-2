from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

class Bandit(ABC):
    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


class EpsilonGreedy(Bandit):
    def __init__(self, p, n_trials=20000):
        self.p = p
        self.n_trials = n_trials
        self.epsilon = 1.0
        self.n = len(p)
        self.counts = [0] * self.n
        self.values = [0.0] * self.n
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return "EpsilonGreedy Bandit"

    def pull(self, arm):
        return 1 if random.random() < self.p[arm] else 0

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def experiment(self):
        optimal_reward = max(self.p) * self.n_trials
        for t in range(1, self.n_trials + 1):
            self.epsilon = 1 / t
            if random.random() < self.epsilon:
                arm = random.randint(0, self.n - 1)
            else:
                arm = np.argmax(self.values)

            reward = self.pull(arm)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.regrets.append(optimal_reward - sum(self.rewards))

    def report(self):
        avg_reward = np.mean(self.rewards)
        avg_regret = np.mean(self.regrets)
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info(f"Average Regret: {avg_regret:.4f}")
        data = {"Bandit": range(len(self.rewards)), "Reward": self.rewards, "Algorithm": ["EpsilonGreedy"] * len(self.rewards)}
        pd.DataFrame(data).to_csv("epsilon_greedy_rewards.csv", index=False)


class ThompsonSampling(Bandit):
    def __init__(self, p, n_trials=20000):
        self.p = p
        self.n_trials = n_trials
        self.n = len(p)
        self.alpha = [1] * self.n
        self.beta = [1] * self.n
        self.rewards = []
        self.regrets = []

    def __repr__(self):
        return "ThompsonSampling Bandit"

    def pull(self, arm):
        return 1 if random.random() < self.p[arm] else 0

    def update(self, arm, reward):
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self):
        optimal_reward = max(self.p) * self.n_trials
        for _ in range(self.n_trials):
            theta = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n)]
            arm = np.argmax(theta)
            reward = self.pull(arm)
            self.update(arm, reward)
            self.rewards.append(reward)
            self.regrets.append(optimal_reward - sum(self.rewards))

    def report(self):
        avg_reward = np.mean(self.rewards)
        avg_regret = np.mean(self.regrets)
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info(f"Average Regret: {avg_regret:.4f}")
        data = {"Bandit": range(len(self.rewards)), "Reward": self.rewards, "Algorithm": ["ThompsonSampling"] * len(self.rewards)}
        pd.DataFrame(data).to_csv("thompson_sampling_rewards.csv", index=False)


class Visualization:
    def plot1(self, rewards_eg, rewards_ts):
        plt.plot(np.cumsum(rewards_eg), label="Epsilon Greedy")
        plt.plot(np.cumsum(rewards_ts), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.title("Cumulative Rewards")
        plt.show()

    def plot2(self, regrets_eg, regrets_ts):
        plt.plot(np.cumsum(regrets_eg), label="Epsilon Greedy")
        plt.plot(np.cumsum(regrets_ts), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.title("Cumulative Regrets")
        plt.show()


def comparison():
    bandit_rewards = [0.1, 0.2, 0.3, 0.4]
    eg_bandit = EpsilonGreedy(bandit_rewards)
    ts_bandit = ThompsonSampling(bandit_rewards)

    logger.info("Running Epsilon Greedy Bandit")
    eg_bandit.experiment()
    eg_bandit.report()

    logger.info("Running Thompson Sampling Bandit")
    ts_bandit.experiment()
    ts_bandit.report()

    viz = Visualization()
    viz.plot1(eg_bandit.rewards, ts_bandit.rewards)
    viz.plot2(eg_bandit.regrets, ts_bandit.regrets)


if __name__ == "__main__":
    comparison()
