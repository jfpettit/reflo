# rlpack-tf

This repository contains implementations of some reinforcement learning algorithms. The pytorch version of rlpack is [here](https://github.com/jfpettit/rl-pack) but it is targeted mainly at simplicity and clarity so that beginners can easily understand the code. rlpack-tf focuses more on performance, but also still tries to maintain readability. Much of the code is inspired by OpenAI's SpinningUp [course](https://spinningup.openai.com/en/latest/index.html). In the future, I'll update it for [TensorFlow 2.0](https://www.tensorflow.org/) and for the resulting update of DeepMind's Sonnet [library](https://sonnet.readthedocs.io/en/latest/).

Currently this code includes implementations of [Advantage Actor Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/) and [Proximal Policy Optimization (PPO)](https://openai.com/blog/openai-baselines-ppo/). The SpinningUp page also includes clear, concise algorithm explanations. 

In the future, I'll likely extend this library to include [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971), [Twin Delayed DDPG (TD3)](https://spinningup.openai.com/en/latest/algorithms/td3.html) and [Soft Actor Critic (SAC)](https://spinningup.openai.com/en/latest/algorithms/sac.html). However, there is no schedule for when these algorithms will be released. 

## Installation

You can install rlpack-tf with the following:

```
git clone https://github.com/jfpettit/rlpack-tf.git
cd rlpack-tf
pip install -e .
```

This install method will also install of the requirements listed below. It is recommended to use a virtual env when installing this code, to avoid issues that might come from different package versions.

rlpack-tf has the following requirements:
- [NumPy](https://numpy.org/)
- [TensorFlow](https://tensorflow.org/)
- [Gym](https://gym.openai.com/)
- [SciPy](https://www.scipy.org/)
- [DeepMind Sonnet](https://sonnet.readthedocs.io/en/latest/)
- [TensorFlow Probability](https://www.tensorflow.org/probability/)
- [Pandas](https://pandas.pydata.org/)
- [MatPlotLib](https://matplotlib.org/)
- [Roboschool](https://github.com/openai/roboschool)

The Gym and Roboschool requirements will be needed to run example files that'll be added in the future. Currently, Pandas and MatPlotLib aren't really used in the package, but will be when I add an automatic plotting utility. 

## Usage

rlpack-tf is intended to be used in python files or in Jupyter notebooks. The API is quite simple:

```python
import gym
import roboschool
from rlpacktf import ppo

env = gym.make('RoboschoolInvertedPendulum-v1')
learner = ppo.PPO(env)
train_ep_returns, train_ep_lens = learner.learn()
```

You can then visualize the outputs. the ```.learn()``` function also takes some other arguments. Currently, these are best discovered by looking at the code. In the future, I'll add thorough examples demonstrating usage of the package and, if there is demand, will produce a docs page.

The default policy and value functions are MLPs from the Sonnet library. If you'd like to write your own custom MLP, you can, but it must follow the same API that the Sonnet MLP does, [described here](https://sonnet.readthedocs.io/en/latest/api.html#mlp).

## Future Plans
- Add model and results saving utility
- Add plotting utility
- Benchmark A2C and PPO on a suite of tasks
- Implement other algorithms listed above
- Blog post
