import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from rlpacktf import utils
from gym.spaces import Discrete, Box
import sonnet as snt
from rlpacktf.a2c import A2C


class PPO(A2C):
    def __init__(self, env, actor=snt.nets.MLP, critic=snt.nets.MLP, actor_critic_hidden_sizes=[32, 32], epoch_interactions=4000, epochs=50,
                 gamma=0.99, policy_learning_rate=3e-4, valuef_learning_rate=1e-3, valuef_train_iters=80, policy_train_iters=80, lam=0.97,
                 max_episode_length=1000, epsilon=0.2, max_kl=0.01):

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        self.observation_ph, action_ph = utils.placeholders_from_spaces(
            env.observation_space, env.action_space)
        advantage_ph, return_ph, logprobs_old_ph = utils.placeholders(
            None, None, None)

        value_f = critic(actor_critic_hidden_sizes +
                         [1], activation=tf.nn.tanh, activate_final=False)
        self.state_values = value_f(self.observation_ph)

        if isinstance(env.action_space, Discrete):
            policy = actor(actor_critic_hidden_sizes +
                           [env.action_space.n], activation=tf.nn.tanh, activate_final=False)
            logits = policy(self.observation_ph)
            logprobs_all = tf.nn.log_softmax(logits)
            action = tf.squeeze(tf.multinomial(logits, 1), axis=1)
            logprobs = tf.reduce_sum(tf.one_hot(
                action_ph, depth=env.action_space.n) * logprobs_all, axis=1)
            logprobs_action = tf.reduce_sum(tf.one_hot(
                action, depth=env.action_space.n) * logprobs_all, axis=1)

        elif isinstance(env.action_space, Box):
            policy = actor(actor_critic_hidden_sizes +
                           [env.action_space.shape[0]], activation=tf.nn.tanh, activate_final=False)
            act_dim = advantage_ph.shape.as_list()[-1]
            mu = policy(self.observation_ph)
            log_std = tf.convert_to_tensor(-0.5*np.ones(act_dim,
                                                        dtype=np.float32), dtype=tf.float32)
            std = tf.cast(tf.exp(log_std), dtype=tf.float32)
            action = mu + tf.random_normal(tf.shape(mu)) * std
            logprobs = utils.calc_log_probs(mu, log_std, action_ph)
            logprobs_action = utils.calc_log_probs(mu, log_std, action)

        self.all_phs = [self.observation_ph, action_ph,
                        advantage_ph, return_ph, logprobs_old_ph]

        self.get_action = [action, self.state_values, logprobs_action]

        self.experience_buffer = utils.Buffer(
            observation_shape, action_shape, epoch_interactions, gamma=gamma, lam=lam)

        pol_ratio = tf.exp(logprobs - logprobs_old_ph)
        clip_pol_ratio = tf.clip_by_value(pol_ratio, 1-epsilon, 1+epsilon)
        self.policy_loss = -tf.reduce_mean(tf.minimum(
            pol_ratio*advantage_ph, clip_pol_ratio*advantage_ph))
        self.value_loss = tf.reduce_mean((return_ph - self.state_values)**2)

        self.approx_kl = tf.reduce_mean(logprobs_old_ph - logprobs)
        self.approx_entropy = tf.reduce_mean(-logprobs)

        self.policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=policy_learning_rate).minimize(self.policy_loss)
        self.valuef_optimizer = tf.train.AdamOptimizer(
            learning_rate=valuef_learning_rate).minimize(self.value_loss)

        self.valuef_train_iters = valuef_train_iters
        self.policy_train_iters = policy_train_iters
        self.max_episode_length = max_episode_length
        self.epoch_interactions = epoch_interactions
        self.max_kl = max_kl

        self.env = env

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        graph_inputs = {x: y for x, y in zip(
            self.all_phs, self.experience_buffer.gather())}
        old_policy_loss, old_value_loss, entropy = self.sess.run(
            [self.policy_loss, self.value_loss, self.approx_entropy], feed_dict=graph_inputs)

        for iter_ in range(self.policy_train_iters):
            _, kl = self.sess.run(
                [self.policy_optimizer, self.approx_kl], feed_dict=graph_inputs)
            if kl > 1.5 * self.max_kl:
                print('Reached max_kl at step {}. Early stopping.'.format(iter_))
                break

        for _ in range(self.valuef_train_iters):
            self.sess.run(self.valuef_optimizer, feed_dict=graph_inputs)

        new_policy_loss, new_value_loss, kl_divergence = self.sess.run(
            [self.policy_loss, self.value_loss, self.approx_kl], feed_dict=graph_inputs)

        return new_policy_loss, new_value_loss, kl_divergence, entropy
