import tensorflow as tf
import tensorflow.keras as k
import numpy as np
#import matplotlib.pyplot as plt
import gym
import roboschool
import pandas as pd
from a2c import A2C
from ppo import PPO
import sonnet as snt

if __name__ == '__main__':
    env = gym.make('RoboschoolHumanoid-v1')
    #print('ENV ACTION SPACE', env.action_space.n, '\n')

    ppo = PPO(env)
    ppo.learn(epochs=250)
