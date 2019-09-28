import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from rlpacktf import utils, mpi_utils
from gym.spaces import Discrete, Box
from gym import wrappers
import sonnet as snt


class A2C:
    def __init__(self, env, actor=snt.nets.MLP, critic=snt.nets.MLP, actor_critic_hidden_sizes=[32, 32], epoch_interactions=4000,
                 epochs=50, gamma=0.99, policy_learning_rate=3e-4, valuef_learning_rate=1e-3, valuef_train_iters=80, lam=0.97, 
                 max_episode_length=1000, track_run=False, track_dir=None, plot_when_done=False):

        self.plot_when_done = plot_when_done
        if self.plot_when_done:
            self.plotter = utils.Plotter()

        self.track_run = track_run
        self.track_dir = track_dir

        if track_run:
            assert track_dir is not None, 'In order to track the run you must provide a directory to save run details to.'
            self.tracker = utils.Tracker(track_dir)
            self.tracker.add_metrics('mean_episode_return', 'mean_episode_length', 'std_episode_return')

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        self.observation_ph, action_ph = utils.placeholders_from_spaces(
            env.observation_space, env.action_space)
        advantage_ph, return_ph, logprobs_old_ph = utils.placeholders(
            None, None, None)

        value_f = critic(actor_critic_hidden_sizes +
                         [1], activation=tf.nn.tanh, activate_final=False, name="value_f")
        self.state_values = value_f(self.observation_ph)

        if isinstance(env.action_space, Discrete):
            policy = actor(actor_critic_hidden_sizes +
                           [env.action_space.n], activation=tf.nn.tanh, activate_final=False, name="policy")
            logits = policy(self.observation_ph)
            logprobs_all = tf.nn.log_softmax(logits)
            action = tf.squeeze(tf.multinomial(logits, 1), axis=1)
            logprobs = tf.reduce_sum(tf.one_hot(
                action_ph, depth=env.action_space.n) * logprobs_all, axis=1)
            logprobs_action = tf.reduce_sum(tf.one_hot(
                action, depth=env.action_space.n) * logprobs_all, axis=1)

        elif isinstance(env.action_space, Box):
            policy = actor(actor_critic_hidden_sizes +
                           [env.action_space.shape[0]], activation=tf.nn.tanh, activate_final=False, name="policy")
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

        self.policy_loss = -tf.reduce_mean(logprobs * advantage_ph)
        self.value_loss = tf.reduce_mean((return_ph - self.state_values)**2)

        self.approx_kl = tf.reduce_mean(logprobs_old_ph - logprobs)
        self.approx_entropy = tf.reduce_mean(-logprobs)

        self.policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=policy_learning_rate).minimize(self.policy_loss)
        self.valuef_optimizer = tf.train.AdamOptimizer(
            learning_rate=valuef_learning_rate).minimize(self.value_loss)

        self.valuef_train_iters = valuef_train_iters
        self.max_episode_length = max_episode_length
        self.epoch_interactions = epoch_interactions

        self.env = env

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        graph_inputs = {x: y for x, y in zip(
            self.all_phs, self.experience_buffer.gather())}
        old_policy_loss, old_value_loss, entropy = self.sess.run(
            [self.policy_loss, self.value_loss, self.approx_entropy], feed_dict=graph_inputs)

        self.sess.run(self.policy_optimizer, feed_dict=graph_inputs)

        for _ in range(self.valuef_train_iters):
            self.sess.run(self.valuef_optimizer, feed_dict=graph_inputs)

        new_policy_loss, new_value_loss, kl_divergence = self.sess.run(
            [self.policy_loss, self.value_loss, self.approx_kl], feed_dict=graph_inputs)

        return new_policy_loss, new_value_loss, kl_divergence, entropy

    def learn(self, epochs=50, render_epochs=None, render_frames=250, save_renders=False):

        obs, reward, done, ep_return, ep_length = self.env.reset(), 0, False, 0, 0

        all_train_ep_rews = []
        all_train_ep_lens = []
        for epoch in range(epochs):
            epochrew = []
            epochlen = []
            for step in range(self.epoch_interactions):
                action, value, logprobs_step = self.sess.run(
                    self.get_action, feed_dict={self.observation_ph: obs.reshape(1, -1)})

                self.experience_buffer.push(
                    obs, action, reward, value, logprobs_step)

                obs, reward, done, _ = self.env.step(action[0])
                ep_return += reward
                ep_length += 1

                over = done or (ep_length == self.max_episode_length)
                if over or (step == self.epoch_interactions-1):
                    if not over:
                        print(
                            'Warning: trajectory cut off by epoch at {} steps'.format(ep_length))

                    last_value = reward if done else self.sess.run(
                        self.state_values, feed_dict={self.observation_ph: obs.reshape(1, -1)})
                    self.experience_buffer.end_trajectory(
                        last_value=last_value)
                    if done:
                        epochlen.append(ep_length)
                        epochrew.append(ep_return)
                        all_train_ep_lens.append(ep_length)
                        all_train_ep_rews.append(ep_return)

                    obs, reward, done, ep_return, ep_length = self.env.reset(), 0, False, 0, 0

            pol_loss, val_loss, kl, entropy = self.update()

            print(
                '-------------------------------\n'
                'Epoch {} of {}\n'.format(epoch, epochs-1),
                'EpReturnMean: {}\n'.format(np.mean(epochrew)),
                'EpReturnStdDev: {}\n'.format(np.std(epochrew)),
                'EpReturnMax: {}\n'.format(np.max(epochrew)),
                'EpReturnMin: {}\n'.format(np.min(epochrew)),
                'EpLenMean: {}\n'.format(np.mean(epochlen)),
                'PolicyEntropy: {}\n'.format(entropy),
                'PolicyLoss: {}\n'.format(pol_loss),
                'ValueLoss: {}\n'.format(val_loss),
                'ApproxKL: {}\n'.format(kl),
                '\n', end='')

            if self.track_run:
                self.tracker.update_metrics(mean_episode_return=np.mean(epochrew), mean_episode_length=np.mean(epochlen),
                    std_episode_return=np.std(epochrew))
                self.tracker.save_metrics()

            if render_epochs is not None and epoch in render_epochs:
                self.watch_model(render_frames=render_frames, save_renders=save_renders, epoch=epoch)
        
        if self.plot_when_done:
            self.plotter.plot(all_train_ep_rews)

        return all_train_ep_rews, all_train_ep_lens

    def watch_model(self, render_frames=250, save_renders=False, epoch=None):
        if save_renders:
            if self.track_dir:
                path=self.track_dir+'/'+self.env.unwrapped.spec.id+str(epoch)
            else:
                path=self.env.unwrapped.spec.id+str(epoch)
            local_env = wrappers.Monitor(env, path, 
                video_callable=lambda episode_id: True, force=True)
        else:
            local_env = self.env

        obs = local_env.reset()
        for frame in range(render_frames):
            action, value, logprobs_step = self.sess.run(
                self.get_action, feed_dict={self.observation_ph: obs.reshape(1, -1)})
            obs, rew, done, _ = self.env.step(action[0])
            local_env.render()
            if done:
                obs = local_env.reset()
        local_env.close()

