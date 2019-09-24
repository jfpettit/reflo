import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import gym
import roboschool
import pandas as pd
from rlpacktf.a2c import A2C
from rlpacktf.ppo import PPO
import sonnet as snt

if __name__ == '__main__':
    env = gym.make('RoboschoolInvertedPendulum-v1')

    ppo = PPO(env, plot_when_done=True)
    ppo.learn(epochs=10)
