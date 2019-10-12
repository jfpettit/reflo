import numpy as np
import tensorflow as tf
from rlpacktf import utils, mpi_utils, logging
from gym.spaces import Discrete, Box
from gym import wrappers
import sonnet as snt
import time


class A2C:
    def __init__(self, env_func, actor=snt.nets.MLP, critic=snt.nets.MLP, actor_critic_hidden_sizes=[32, 32], epoch_interactions=4000,
                 gamma=0.99, policy_learning_rate=3e-4, valuef_learning_rate=1e-3, valuef_train_iters=80, lam=0.97,
                 max_episode_length=1000, track_run=False, track_dir=None, plot_when_done=False, logger_fname=None):

        self.init_common(env_func=env_func, actor=actor, critic=critic,
                         actor_critic_hidden_sizes=actor_critic_hidden_sizes, epoch_interactions=epoch_interactions,
                         gamma=gamma, policy_learning_rate=policy_learning_rate, valuef_learning_rate=valuef_learning_rate,
                         valuef_train_iters=valuef_train_iters, lam=lam, max_episode_length=max_episode_length, track_run=track_run,
                         track_dir=track_dir, plot_when_done=plot_when_done, logger_fname=logger_fname)

    def init_common(self, env_func, actor, critic, actor_critic_hidden_sizes, epoch_interactions,
                    gamma, policy_learning_rate, valuef_learning_rate, valuef_train_iters, lam, max_episode_length, 
                    track_run, track_dir, plot_when_done, logger_fname):

        env = env_func()

        self.plot_when_done = plot_when_done
        if self.plot_when_done:
            self.plotter = utils.Plotter()

        self.track_run = track_run
        self.track_dir = track_dir
        self.logger = logging.EpochLogger(self.track_dir, logger_fname)

        if track_run:
            assert track_dir is not None, 'In order to track the run you must provide a directory to save run details to.'
            self.tracker = utils.Tracker(track_dir)
            self.tracker.add_metrics(
                'mean_episode_return', 'mean_episode_length', 'std_episode_return')

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape

        self.observation_ph, action_ph = utils.placeholders_from_spaces(
            env.observation_space, env.action_space)
        advantage_ph, return_ph, logprobs_old_ph = utils.placeholders(
            None, None, None)

        value_f = critic(actor_critic_hidden_sizes +
                         [1], activation=tf.nn.tanh, activate_final=False, name='value_f')
        self.state_values = value_f(self.observation_ph)

        if isinstance(env.action_space, Discrete):
            policy = actor(actor_critic_hidden_sizes +
                           [env.action_space.n], activation=tf.nn.tanh, activate_final=False, name='policy')
            logits = policy(self.observation_ph)
            logprobs_all = tf.nn.log_softmax(logits)
            action = tf.squeeze(tf.multinomial(logits, 1), axis=1)
            logprobs = tf.reduce_sum(tf.one_hot(
                action_ph, depth=env.action_space.n) * logprobs_all, axis=1)
            logprobs_action = tf.reduce_sum(tf.one_hot(
                action, depth=env.action_space.n) * logprobs_all, axis=1)

        elif isinstance(env.action_space, Box):
            policy = actor(actor_critic_hidden_sizes +
                           [env.action_space.shape[0]], activation=tf.nn.tanh, activate_final=False, name='policy')
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

        self.approx_kl = tf.reduce_mean(logprobs_old_ph - logprobs)
        self.approx_entropy = tf.reduce_mean(-logprobs)

        self.init_loss_funcs(logprobs=logprobs, advantage_ph=advantage_ph, return_ph=return_ph,
                             policy_learning_rate=policy_learning_rate, valuef_learning_rate=valuef_learning_rate,
                             logprobs_old_ph=logprobs_old_ph)

        self.valuef_train_iters = valuef_train_iters
        self.max_episode_length = max_episode_length
        self.local_epoch_interactions = epoch_interactions // mpi_utils.num_processes()
        self.epoch_interactions = epoch_interactions

        self.experience_buffer = utils.Buffer(
            observation_shape, action_shape, self.local_epoch_interactions, gamma=gamma, lam=lam)

        self.env = env

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(mpi_utils.sync_all_parameters())

    def init_loss_funcs(self, **kwargs):
        self.policy_loss = - \
            tf.reduce_mean(kwargs['logprobs'] * kwargs['advantage_ph'])
        self.value_loss = tf.reduce_mean(
            (kwargs['return_ph'] - self.state_values)**2)

        self.policy_optimizer = mpi_utils.MPIAdamOptimizer(
            learning_rate=kwargs['policy_learning_rate']).minimize(self.policy_loss)
        self.valuef_optimizer = mpi_utils.MPIAdamOptimizer(
            learning_rate=kwargs['valuef_learning_rate']).minimize(self.value_loss)

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

        self.logger.store(PolicyLoss=old_policy_loss, ValueLoss=old_value_loss, ApproxKL=kl_divergence,
                          PolicyEntropy=entropy)

        return new_policy_loss, new_value_loss, kl_divergence, entropy

    def learn(self, epochs=50, render_epochs=None, render_frames=250, save_renders=False, save_policies=False, ncpu=1):

        mpi_utils.mpi_fork(ncpu)

        last_save_ret = -np.inf

        if save_policies:
            assert self.track_dir is not None, 'Must provide directory to log policies to.'

        obs, reward, done, ep_return, ep_length = self.env.reset(), 0, False, 0, 0

        all_train_ep_rews = []
        start_time = time.time()
        for epoch in range(epochs):
            epochrew = []
            epochlen = []
            for step in range(self.local_epoch_interactions):
                action, value, logprobs_step = self.sess.run(
                    self.get_action, feed_dict={self.observation_ph: obs.reshape(1, -1)})
                self.logger.store(StateVals=value)

                self.experience_buffer.push(
                    obs, action, reward, value, logprobs_step)

                obs, reward, done, _ = self.env.step(action[0])
                ep_return += reward
                ep_length += 1

                over = done or (ep_length == self.max_episode_length)
                if over or (step == self.local_epoch_interactions-1):
                    if not over:
                        self.logger.log_msg(
                            'Warning: trajectory cut off by epoch at {} steps'.format(
                                ep_length),
                            color='yellow')

                    last_value = reward if done else self.sess.run(
                        self.state_values, feed_dict={self.observation_ph: obs.reshape(1, -1)})
                    self.experience_buffer.end_trajectory(
                        last_value=last_value)
                    if over:
                        epochlen.append(ep_length)
                        epochrew.append(ep_return)
                        all_train_ep_rews.append(ep_return)
                        self.logger.store(EpReturn=ep_return,
                                          AvgEpLength=ep_length)
                    obs, reward, done, ep_return, ep_length = self.env.reset(), 0, False, 0, 0

            pol_loss, val_loss, kl, entropy = self.update()

            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpReturn', with_min_and_max=True)
            self.logger.log_tabular('AvgEpLength', average_only=True)
            self.logger.log_tabular('StateVals', with_min_and_max=True)
            self.logger.log_tabular(
                'TotalEnvInteracts', (epoch+1)*self.epoch_interactions)
            self.logger.log_tabular('PolicyLoss', average_only=True)
            self.logger.log_tabular('ValueLoss', average_only=True)
            self.logger.log_tabular('PolicyEntropy', average_only=True)
            self.logger.log_tabular('ApproxKL', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()

            if self.track_run:
                self.tracker.update_metrics(mean_episode_return=np.mean(epochrew), mean_episode_length=np.mean(epochlen),
                                            std_episode_return=np.std(epochrew))
                self.tracker.save_metrics()

            if save_policies and np.mean(epochrew) > last_save_ret:
                last_save_ret = np.mean(epochrew)
                utils.save_network(self.policy, self.track_dir,
                                   self.env.unwrapped.spec.id)

            if render_epochs is not None and epoch in render_epochs:
                self.watch_model(render_frames=render_frames,
                                 save_renders=save_renders, epoch=epoch)

        if self.plot_when_done:
            self.plotter.plot(all_train_ep_rews)

        return all_train_ep_rews

    def watch_model(self, render_frames=250, save_renders=False, epoch=None):
        if save_renders:
            if self.track_dir:
                path = self.track_dir+'/'+self.env.unwrapped.spec.id+str(epoch)
            else:
                path = 'tmp'+str(time.time())+'/'+self.env.unwrapped.spec.id+str(epoch)
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
