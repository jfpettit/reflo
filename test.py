import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import gym
import roboschool
import pandas as pd
from rlpacktf.a2c import A2C
from rlpacktf.ppo import PPO
from rlpacktf import mpi_utils
import sonnet as snt

if __name__ == '__main__':
    agent = PPO(lambda: gym.make('RoboschoolInvertedPendulum-v1'), track_dir='test_logging')
    agent.learn(epochs=200, ncpu=2, save_policies=True)
