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
    mpi_utils.mpi_fork(2)
    ppo = PPO(lambda: gym.make('RoboschoolAnt-v1'))
    ppo.learn(epochs=200, render_epochs=[199], render_frames=1000, ncpu=2)
