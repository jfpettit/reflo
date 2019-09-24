import numpy as np
import tensorflow as tf
from scipy.signal import lfilter
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt
import pandas as pd
import os

class Plotter:
    def __init__(self, dark=True):
        if dark:
            plt.style.use("seaborn-darkgrid")

    def plot(self, data, key='mean_episode_return'):
        data = pd.Series(data).rolling(window=len(data)//10).mean()
        datastd = pd.Series(data).rolling(window=len(data)//10).std()
        plt.plot(data)
        plt.title(key+' per episode')
        plt.xlabel('episodes')
        plt.ylabel(key)
        plt.fill_between(range(len(data)), data+datastd, data-datastd, alpha=0.2)
        plt.show()


class Tracker:
    def __init__(self, track_dir):
        self.track_dict = {}
        self.track_dir = track_dir

    def add_metrics(self, *args):
        for arg in args:
            self.track_dict[arg] = []

    def update_metrics(self, **kwargs):
        for key, value in kwargs.items():
            self.track_dict[key].append(value)

    def save_metrics(self):
        if not os.path.isdir(self.track_dir):
            os.mkdir(self.track_dir)
        for key in list(self.track_dict.keys()):
            np.save(self.track_dir+'/'+key, self.track_dict[key])

class Buffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.97):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam

        self.state_record = np.zeros(
            combined_shape(size, state_dim), dtype=np.float32)
        self.action_record = np.zeros(
            combined_shape(size, action_dim), dtype=np.float32)
        self.advantage_record = np.zeros(size, dtype=np.float32)
        self.return_record = np.zeros(size, dtype=np.float32)
        self.logprobs_record = np.zeros(size, dtype=np.float32)
        self.rew_record = np.zeros(size, dtype=np.float32)
        self.value_record = np.zeros(size, dtype=np.float32)

        self.point_idx, self.start_idx = 0, 0

    def push(self, state, action, reward, value, logprob):
        assert self.point_idx <= self.size

        self.state_record[self.point_idx] = state
        self.action_record[self.point_idx] = action
        self.rew_record[self.point_idx] = reward
        self.value_record[self.point_idx] = value
        self.logprobs_record[self.point_idx] = logprob

        self.point_idx += 1

    def discount_cumulative_sum(self, x, discount):
        return lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def end_trajectory(self, last_value=0):
        traj_slice = slice(self.start_idx, self.point_idx)

        rews = self.rew_record[traj_slice]
        vals = np.append(self.value_record[traj_slice], last_value)

        deltas = rews + self.gamma * vals[1:] - vals[:-1]

        self.advantage_record[traj_slice] = self.discount_cumulative_sum(
            deltas, self.gamma * self.lam)

        self.return_record[traj_slice] = self.discount_cumulative_sum(
            rews, self.gamma)

        self.start_idx = self.point_idx

    def gather(self):
        assert self.point_idx == self.size

        self.point_idx, self.start_idx = 0, 0
        self.advantage_record = (self.advantage_record - np.mean(
            self.advantage_record))/(np.std(self.advantage_record) + 1e-8)

        return [self.state_record, self.action_record, self.advantage_record, self.return_record, self.logprobs_record]


def calc_log_probs(action_mu, action_logstd, action):
    lp = -0.5 * (((action-action_mu)/(tf.exp(action_logstd)+1e-8))
                 ** 2 + 2*action_logstd + tf.log(2*tf.constant(np.pi, dtype=tf.float32)))
    return tf.reduce_sum(lp, axis=1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]
