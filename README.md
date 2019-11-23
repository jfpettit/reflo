# reflo

**Note: there is an issue with the MPI parallelization in this code. Running an algorithm on a single CPU appears to work and yield satisfactory performance. However, when the number of CPUs is set to more than one, performance degrades. I'm working on a solid confirmation that the single-CPU code is correct and then will find and fix the error with running across multiple CPUs.**

This repository contains implementations of some reinforcement learning algorithms. The pytorch version of rlpack is [here](https://github.com/jfpettit/rl-pack) but it is targeted mainly at simplicity and clarity so that beginners can easily understand the code. reflo focuses more on performance, but also still tries to maintain readability. Much of the code is inspired by OpenAI's SpinningUp [course](https://spinningup.openai.com/en/latest/index.html). 

Currently this code includes implementations of [Advantage Actor Critic (A2C)](https://openai.com/blog/baselines-acktr-a2c/) and [Proximal Policy Optimization (PPO)](https://openai.com/blog/openai-baselines-ppo/). The SpinningUp page also includes clear, concise algorithm explanations.

In the future, I'll likely extend this library to include [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971), [Twin Delayed DDPG (TD3)](https://spinningup.openai.com/en/latest/algorithms/td3.html) and [Soft Actor Critic (SAC)](https://spinningup.openai.com/en/latest/algorithms/sac.html). However, there is no schedule for when these algorithms will be released.

## Installation

You can install reflo with the following:

```
git clone https://github.com/jfpettit/reflo.git
cd reflo
pip install -e .
```

This install method will also install of the requirements listed below. It is recommended to use a virtual env when installing this code, to avoid issues that might come from different package versions.

reflo has the following requirements:
- [NumPy](https://numpy.org/)
- [TensorFlow 1.15](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)
- [Gym](https://gym.openai.com/)
- [SciPy](https://www.scipy.org/)
- [DeepMind Sonnet](https://sonnet.readthedocs.io/en/latest/)
- [TensorFlow Probability](https://www.tensorflow.org/probability/)
- [Pandas](https://pandas.pydata.org/)
- [MatPlotLib](https://matplotlib.org/)
- [Roboschool](https://github.com/openai/roboschool): Now deprecated by OpenAI Gym in favor of [pybullet envs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)
- [MPI for Python](https://mpi4py.readthedocs.io/en/stable/)
- [PyBullet envs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)

The Gym and Roboschool requirements will be needed to run example files that'll be added in the future. Note that mpi4py will require that you have a working MPI installed on your machine. Here are installation guides for a [Mac](http://www.science.smith.edu/dftwiki/index.php/Install_MPI_on_a_MacBook), and for [Windows](https://nyu-cds.github.io/python-mpi/setup/).

## Usage
reflo can be used in the command line or in python files/Jupyter notebooks. To use it in the command line:

```
cd reflo
python runner.py -h
```

This will list the full set of optional arguments for the command line runner. Running ```python runner.py``` will run with the default set of arguments.

To use it in a Python file or Jupyter notebook, the API is quite simple:

```python
import gym
import roboschool
from reflo import ppo

agent = ppo.PPO(lambda : gym.make('RoboschoolInvertedPendulum-v1')) # or other Gym env
train_ep_returns, train_ep_lens = agent.learn()
```

You can then visualize the outputs.

The PPO and A2C classes also have some arguments, mostly optional. They share the majority of these but PPO has a couple of arguments that A2C does not.

The shared arguments; an argument is optional unless specified otherwise:
- ```env_func```: Not optional. A function initializing the environment you'd like to run.
- ```actor```: a function which takes in a list of integers and returns a neural network with hidden layer sizes from that list of integers. Default: ```sonnet.nets.MLP```.
- ```critic```: a function which takes in a list of integers and returns a neural network with hidden layer sizes from that list of integers. Default: ```sonnet.nets.MLP```.
- ```actor_critic_hidden_sizes```: a list of numbers representing the hidden layer sizes for the actor and critic. Default: ```[32, 32]```.
- ```epoch_interactions```: interaction steps taken per epoch. Default: 4000.
- ```gamma```: reward discount factor. Default: 0.99.
- ```policy_learning_rate```: learning rate of the policy optimizer. Default: ```3e-4```.
- ```valuef_learning_rate```: learning rate of the value function optimizer. Default: ```1e-3```.
- ```valuef_train_iters```: number of training steps for the value function per epoch. Default: 80
- ```lam```: lambda from GAE-lambda advantage estimation. Default: 0.97.
- ```max_episode_length```: maximum number of interactions allowed in an episode. Default: 1000.
- ```track_run```: Boolean, whether to save data tracked during the run into .npy files. Default: False.
- ```track_dir```: String, directory to save the run data to. Required if ```track_run``` set to true. Default: None.
- ```plot_when_done```: Boolean, whether to plot a reward curve when training is finished. Default: False.
- ```logger_fname```: String, filename to output logger dumps to. It uses ```track_dir``` for the directory to store this file in. Default: None.

Following are a couple of arguments that are PPO specific:
- ```policy_train_iters```: Integer, training steps per epoch to do for the policy. Default = 80
- ```epsilon```: Float, value in the loss function to restrict how far the new policy gets from the old. Default = 0.2.
- ```max_kl```: Float, maximum KL divergence allowed between the new policy and the old. Default = 0.01.

The ```.learn()``` function also takes the following optional arguments:
- ```epochs```: Integer, number of epochs to train for. Default: 50.
- ```render_epochs```: List of integers, epochs on which to watch the policy interact with the environment. Default: None.
- ```render_frames```: Integer, number of interaction frames to watch. Default: 250.
- ```save_renders```: Boolean, whether or not to save the frames of agent-environment interaction. Default: False.
- ```save_session```: Boolean, whether to save the TF session over training. Only saves if the updated policy performs better than the previous one. Default: False.
- ```ncpu```: Integer, number of CPUs to parallelize the training over. Default: 1.

Using ```save_session``` will dump the TF session to the ```track_dir```. There is not yet a restore function, however, the session should be simple enough to restore by using Tensorflow's ```tf.train.Saver``` functionality.

In the future, I'll add thorough examples demonstrating usage of the package and, if there is demand, will produce a docs page.

The default policy and value functions are MLPs from the Sonnet library. If you'd like to write your own custom network, you can, but it must follow the same API that the Sonnet MLP does, [described here](https://sonnet.readthedocs.io/en/latest/api.html#mlp).

This code has not been tested with recurrent networks.

## Future Plans
- Add support for recurrent networks
- Benchmark A2C and PPO on a suite of tasks
- Implement other algorithms listed above
- Blog post
- Implement a session restoring functionality
