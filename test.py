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
    env = gym.make('RoboschoolHalfCheetah-v1')

    ppo = PPO(env, actor_critic_hidden_sizes=[64, 64], plot_when_done=True, max_episode_length=2000, epoch_interactions=8000)
    ppo.learn(epochs=100, render_epochs=[99], render_frames=1000)
