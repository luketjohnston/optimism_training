import os.path as osp
import gym
import sys
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.progress.utils import discount_with_dones
from baselines.progress.utils import Scheduler, make_path, find_trainable_variables
from baselines.progress.policies import CnnPolicy
from baselines.progress.utils import cat_entropy, mse


# TODO tomorrow:
# why is it collapsing to always pick the same action? This shouldn't occur, is way not optimal.

# the working params are 0.1 and 0.02, seed 2 and maybe 1?

# original default was 7e-4
LEARNING_RATE = 7e-4

PROGRESS_REWARD_SCALE = .01 
#PROGRESS_REWARD_SCALE = 0.000001
PROGRESS_LOSS_SCALE = 0.02
#PROGRESS_LOSS_SCALE = 0.002
MY_ENT_COEF = .01 # originally 0.01
HALT_AFTER_REWARD = False


RENDERING=True

test_without_progress = False
if test_without_progress:
  PROGRESS_REWARD_SCALE = 0.0 
  PROGRESS_LOSS_SCALE = 0.0


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=MY_ENT_COEF, vf_coef=0.5, max_grad_norm=0.5, lr=LEARNING_RATE,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        # if you don't want to use progress, set to 0
        progress_target = tf.placeholder(tf.uint8, shape=[nbatch]) #obs
        progress_loss = PROGRESS_LOSS_SCALE * tf.losses.absolute_difference(
            train_model.progress, progress_target)
        # DONT NEED TO DO THE BELOW, included progress loss in rewards in Runner.run
        ## adjust advantage by progress loss, so policy learns to not go
        ## places where it can't predict the progress
        #ADV_adj = ADV - tf.stop_gradient(progress_loss) 
        #progress_loss=0
        

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R)) 
        entropy = tf.reduce_mean(cat_entropy(train_model.pi)) 
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef + progress_loss
        #loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, progress):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, 
                progress_target:progress}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, progress_loss0, _ = sess.run(
                [pg_loss, vf_loss, entropy, progress_loss, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, progress_loss0

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        #nh, nw, nc = env.observation_space.shape
        nh, nw, nc = 1,1,2
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.progress = [0 for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        #self.obs[:, :, :, -self.nc:] = obs
        self.obs[:, :, :, -self.nc:] = np.reshape(obs, (self.obs.shape))

    def run(self, rendering=False):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, = [],[],[],[],[]
        mb_progress, mb_progress_t, mb_real_rewards = [],[],[] # stuff i've added
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, progress = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_progress.append(progress)
            obs, rewards, dones, _ = self.env.step(actions)
            if rendering:
              self.env.render(1)
            mb_real_rewards.append(rewards)
            self.progress = self.progress + (1 - dones) # udpate progress for each environment
            # save progress to be formatted for minibatch later
            mb_progress_t.append(self.progress) 
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    self.progress[n] = 0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_progress = np.asarray(mb_progress, dtype=np.int32).swapaxes(1, 0)
        mb_progress_t = np.asarray(mb_progress_t, dtype=np.int32).swapaxes(1, 0)
        mb_real_rewards = np.asarray(mb_real_rewards, dtype=np.int32).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # compute progress reward. Can only be negative.
        #print(mb_progress_t)
        mb_progress_rewards = (-1 * np.abs(mb_progress - mb_progress_t))
        # update rewards with progress rewards.
        mb_rewards = mb_rewards + PROGRESS_REWARD_SCALE * mb_progress_rewards
        #print(mb_rewards)

        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            # TODO right here I have to add progress rewards
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        mb_progress_t = mb_progress_t.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_real_rewards, mb_progress_t

def learn(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=MY_ENT_COEF, max_grad_norm=0.5, lr=LEARNING_RATE, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    save_dir = 'save/'
    make_path(save_dir)
    save_path = save_dir + 'model.save'
    progress_loss_f = open(save_dir + "progress.txt", "w", 1) # 1 is line buffered


    accumulated_rewards = 0
    for update in range(1, total_timesteps//nbatch+1):
        rendering = ((update // 30) % 50 == 0) and RENDERING
        obs, states, rewards, masks, actions, values, real_rewards, progress = runner.run(rendering)
        accumulated_rewards += np.sum(real_rewards) # only want to record times when we get positive reward
        if np.sum(real_rewards > 0) > 0 and HALT_AFTER_REWARD:
          print("Found first reward after %d updates." % update)
          sys.exit()
        policy_loss, value_loss, policy_entropy, progress_loss = model.train(obs, states, rewards, masks, actions, values, progress)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            progress_loss_f.write("%f\n" % progress_loss)
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("progress_loss", float(progress_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("accumulated rewards", accumulated_rewards)
            logger.dump_tabular()
        save_interval = 4000
        if update % save_interval == 0:
            print("Saving model to %s" % save_path)
            model.save(save_path)
    env.close()
 

if __name__ == '__main__':
    main()