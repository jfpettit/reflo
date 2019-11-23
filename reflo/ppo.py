# pylint: disable=import-error
import numpy as np
import tensorflow as tf
from reflo import utils, mpi_utils, logging
from reflo.a2c import A2C
from gym.spaces import Discrete, Box
import sonnet as snt
import time


class PPO(A2C):
    def __init__(
        self,
        env_func,
        actor=snt.nets.MLP,
        critic=snt.nets.MLP,
        actor_critic_hidden_sizes=[32, 32],
        epoch_interactions=4000,
        gamma=0.99,
        policy_learning_rate=3e-4,
        valuef_learning_rate=1e-3,
        valuef_train_iters=80,
        policy_train_iters=80,
        lam=0.97,
        max_episode_length=1000,
        epsilon=0.2,
        max_kl=0.01,
        track_run=False,
        track_dir=None,
        plot_when_done=False,
        logger_fname=None,
        ncpu=1,
        seed=None,
        train_render=False
    ):

        mpi_utils.mpi_fork(ncpu)

        self.epsilon = epsilon
        self.policy_train_iters = policy_train_iters
        self.max_kl = max_kl

        self.init_common(
            env_func=env_func,
            actor=actor,
            critic=critic,
            actor_critic_hidden_sizes=actor_critic_hidden_sizes,
            epoch_interactions=epoch_interactions,
            gamma=gamma,
            policy_learning_rate=policy_learning_rate,
            valuef_learning_rate=valuef_learning_rate,
            valuef_train_iters=valuef_train_iters,
            lam=lam,
            max_episode_length=max_episode_length,
            track_run=track_run,
            track_dir=track_dir,
            plot_when_done=plot_when_done,
            logger_fname=logger_fname,
            seed=seed,
            train_render=train_render
        )

    def init_loss_funcs(self, **kwargs):
        pol_ratio = tf.exp(kwargs["logprobs"] - kwargs["logprobs_old_ph"])
        clip_pol_ratio = tf.clip_by_value(pol_ratio, 1 - self.epsilon, 1 + self.epsilon)
        self.policy_loss = -tf.reduce_mean(
            tf.minimum(
                pol_ratio * kwargs["advantage_ph"],
                clip_pol_ratio * kwargs["advantage_ph"],
            )
        )
        self.value_loss = tf.reduce_mean((kwargs["return_ph"] - self.state_values) ** 2)

        self.policy_optimizer = mpi_utils.MPIAdamOptimizer(
            learning_rate=kwargs["policy_learning_rate"]
        ).minimize(self.policy_loss)
        self.valuef_optimizer = mpi_utils.MPIAdamOptimizer(
            learning_rate=kwargs["valuef_learning_rate"]
        ).minimize(self.value_loss)

    def update(self):
        graph_inputs = {
            x: y for x, y in zip(self.all_phs, self.experience_buffer.gather())
        }
        old_policy_loss, old_value_loss, entropy = self.sess.run(
            [self.policy_loss, self.value_loss, self.approx_entropy],
            feed_dict=graph_inputs,
        )

        for iter_ in range(self.policy_train_iters):
            _, kl = self.sess.run(
                [self.policy_optimizer, self.approx_kl], feed_dict=graph_inputs
            )
            kl = mpi_utils.mpi_avg(kl)
            if kl > 1.5 * self.max_kl:
                self.logger.log_msg(
                    f"Reached max_kl at step {iter_} of {self.policy_train_iters}. Early stopping.",
                    color="yellow",
                )
                break

        for _ in range(self.valuef_train_iters):
            self.sess.run(self.valuef_optimizer, feed_dict=graph_inputs)

        new_policy_loss, new_value_loss, kl_divergence = self.sess.run(
            [self.policy_loss, self.value_loss, self.approx_kl], feed_dict=graph_inputs
        )
        self.logger.store(
            PolicyLoss=old_policy_loss,
            ValueLoss=old_value_loss,
            ApproxKL=kl_divergence,
            PolicyEntropy=entropy,
        )

        return new_policy_loss, new_value_loss, kl_divergence, entropy
