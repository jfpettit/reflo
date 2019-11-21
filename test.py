import tensorflow as tf
import tensorflow.keras as k
import numpy as np
import gym
import pybullet_envs
import pandas as pd
from rlpacktf.a2c import A2C
from rlpacktf.ppo import PPO
from rlpacktf import mpi_utils
import sonnet as snt
import argparse
import gym_snake_rl

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=100)
parser.add_argument('--epoch_interacts', type=int, help='Agent-env interactions per epoch', default=4000)
parser.add_argument('--env', type=str, help='Env to run in.', default='HalfCheetahBulletEnv-v0')
parser.add_argument('--ncpu', type=int, help='Num of CPUs to train on.', default=1)
parser.add_argument('--alg', type=str, help='PPO or A2C', default='PPO')
parser.add_argument('--horizon', type=int, help='Time horizon for end of episode', default='1000')

if __name__ == '__main__':
    args = parser.parse_args()
    print(f"RUN ARGS {args}")
    env = args.env
    
    if args.alg == 'PPO':
        agent = PPO(lambda: gym.make(env), actor_critic_hidden_sizes=[64, 32],
                epoch_interactions=args.epoch_interacts, ncpu=args.ncpu, max_episode_length=args.horizon)
    else:
        agent = A2C(lambda: gym.make(env), actor_critic_hidden_sizes=[64, 32],
                epoch_interactions=args.epoch_interacts, ncpu=args.ncpu, max_episode_length=args.horizon)
    print(f'Training in {env}')
    agent.learn(epochs=args.epochs)
