import numpy as np


class RunningMeanStd_norm:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
        self.dim = shape

    def update(self, x):
        x = x.reshape([-1, x.shape[-1]])
        if self.n == 0:
            self.mean = np.mean(x, axis=0).reshape([self.dim])
            self.std = np.std(x, axis=0).reshape([self.dim])
        else:
            old_mean = self.mean.copy()
            old_S = self.S.copy()
            self.mean = (old_mean * self.n + np.mean(x, axis=0).reshape([self.dim]) * x.shape[0]) / (
                    self.n + x.shape[0])
            self.S = ((self.n * (old_S + old_mean ** 2) + np.sum(x ** 2, axis=0)) / (self.n + x.shape[0])
                      - self.mean ** 2).reshape([self.dim])
            self.std = np.sqrt(self.S)

        self.n += x.shape[0]

        if self.n == 1:
            self.std = self.mean

    def reset(self):
        self.n = 0
        self.mean = np.zeros(self.dim)
        self.S = np.zeros(self.dim)
        self.std = np.sqrt(self.S)

    def set(self, n, mean, S):
        self.n = n
        self.mean = mean
        self.S = S
        self.std = np.sqrt(self.S)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd_norm(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)

        mean = self.running_ms.mean
        std = self.running_ms.std

        for _ in range(x.ndim - 1):
            mean = mean[np.newaxis, :]
            std = std[np.newaxis, :]

        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RunningMeanStd_reward:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
        self.shape = shape

    def update(self, x):
        if self.n == 0:
            self.mean = x
            self.std = x * np.sign(x)
        else:
            old_mean = self.mean.copy()
            old_S = self.S.copy()
            self.mean = (old_mean * self.n + x) / (self.n + 1)
            self.S = (self.n * (old_S + old_mean ** 2) + x ** 2) / (self.n + 1) - self.mean ** 2
            self.std = np.sqrt(self.S)

        self.n += 1

    def combine(self):
        old_mean = self.mean.copy()
        old_S = self.S.copy()
        S_size = old_mean.size
        self.mean = old_mean.mean() * np.ones_like(old_mean)
        self.S = (np.sum(old_S + old_mean ** 2)/S_size - old_mean.mean() ** 2) * np.ones_like(old_S)
        self.std = np.sqrt(self.S)

    def reset(self):
        self.n = 0
        self.mean = np.zeros(self.shape)
        self.S = np.zeros(self.shape)
        self.std = np.sqrt(self.S)

    def set(self, n, mean, S):
        self.n = n
        self.mean = mean * np.ones(self.shape)
        self.S = S * np.ones(self.shape)
        self.std = np.sqrt(self.S)


class RewardScaling:
    def __init__(self, shape, gamma, stop_update_n=None):
        self.shape = shape  # reward shape=6d
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd_reward(self.shape)
        self.R = np.zeros(self.shape)
        self.stop_update_n = stop_update_n

    def __call__(self, x):
        # 每个环境自己跑
        # self.R = self.gamma * self.R + x
        self.R = x
        if self.running_ms.n < self.stop_update_n:
            self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-3)  # Only divided std
        return x

    def combine(self):
        self.running_ms.combine()

    def set(self, n, mean, S):
        self.running_ms.set(n, mean, S)

    def get_para(self):
        self.combine()
        return [self.running_ms.n, self.running_ms.mean.reshape(-1)[0], self.running_ms.S.reshape(-1)[0]]

    def reset(self):
        # self.running_ms.reset()
        self.R = np.zeros(self.shape)
