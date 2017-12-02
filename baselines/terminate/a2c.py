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

from baselines.terminate.utils import discount_with_dones
from baselines.terminate.utils import Scheduler, make_path, find_trainable_variables
from baselines.terminate.policies import CnnPolicy
from baselines.terminate.utils import cat_entropy, mse

SKIPPING_PROGRESS_GRADS = False

# original default was 7e-4
PROGRESS_LEARNING_RATE = 7e-4
LEARNING_RATE = 7e-4

# ENV IS RANDOM ACTION NOW!!!!

ENV_N = 15

PROGRESS_MIN_STEP = 1.000 # This seems to work well (.001)
PROGRESS_LOSS_SCALE = 1.0 / PROGRESS_MIN_STEP # this seems necessary for stability

zero_progress = False
zero_all_except_progress = True

if zero_progress:
  PROGRESS_LOSS_SCALE = 0.0


POLICY_LOSS_SCALE = 1
PROGRESS_REWARD_SCALE = 1 
#PROGRESS_REWARD_SCALE = 1.0
#PROGRESS_REWARD_SCALE = .01 / PROGRESS_MIN_STEP

MY_ENT_COEF = 0.01 # originally 0.01
#MY_ENT_COEF = 0.00 # originally 0.01
#MY_ENT_COEF = 1 # originally 0.01
#MY_ENT_COEF = 1.00 # originally 0.01
HALT_AFTER_REWARD = False

VF_COEF = .5 # originally 0.5
#VF_COEF = 0.0 # originally 0.5

NEG_LOSS_GRAD = 1.0

if zero_all_except_progress:
  POLICY_LOSS_SCALE = 0.0
  PROGRESS_REWARD_SCALE = 0.0
  MY_ENT_COEF = 0.0
  VF_COEF = 0.0

RENDERING=True

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=MY_ENT_COEF, vf_coef=VF_COEF, max_grad_norm=0.5, lr=LEARNING_RATE,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        PROGRESS_LR = tf.placeholder(tf.float32, [])

        PROGRESS_T = tf.placeholder(tf.float32, [None]) # progress target

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True) # model for previous step


        step_error = train_model.progress - PROGRESS_T
        # use abs cuz want grads for pos and neg to be the same.
        pos_prog_loss = tf.reduce_mean(tf.nn.relu(step_error))
        neg_prog_loss = tf.reduce_mean(tf.nn.relu(-step_error)*NEG_LOSS_GRAD)
        unscaled_prog_loss = pos_prog_loss + neg_prog_loss
        progress_loss = PROGRESS_LOSS_SCALE * unscaled_prog_loss

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac) 
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))  
        entropy = tf.reduce_mean(cat_entropy(train_model.pi)) 
        loss = pg_loss*POLICY_LOSS_SCALE - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        progress_grads = tf.gradients(progress_loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            progress_grads, progress_grad_norm = tf.clip_by_global_norm(progress_grads, max_grad_norm)
        grads = list(zip(grads, params))
        progress_grads = list(zip(progress_grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        progress_trainer = tf.train.RMSPropOptimizer(learning_rate=PROGRESS_LR, decay=alpha, epsilon=epsilon)
        _train_policy = trainer.apply_gradients(grads)
        if not SKIPPING_PROGRESS_GRADS:
          _train_progress = progress_trainer.apply_gradients(progress_grads)
          _train = tf.group(_train_policy, _train_progress)
        else:
          _train = _train_policy

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)
        progress_lr = Scheduler(v=PROGRESS_LEARNING_RATE, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, rewards, masks, actions, values, progress_t):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
                progress_cur_lr = progress_lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, 
                PROGRESS_LR:progress_cur_lr, PROGRESS_T: progress_t}
            policy_loss, value_loss, policy_entropy, progress_loss0, total_loss, _ = sess.run(
                [pg_loss, vf_loss, entropy, progress_loss, loss, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, progress_loss0, total_loss

        def train_only_progress(extra_progress_updates):
            if not extra_progress_updates: return None
            obs, progtar = zip(*extra_progress_updates)
            for step in range(len(obs)):
                progress_cur_lr = progress_lr.value()
            td_map = {train_model.X:obs, PROGRESS_LR:progress_cur_lr, PROGRESS_T:progtar}
            progress_loss0, _ = sess.run(
                [progress_loss, _train_progress],
                td_map
            )
            return progress_loss0

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

        def display(mode='progress'):
            target = self.step_model.progress if mode=='progress' else self.step_model.value
            n = min(ENV_N, 20)
            batch = np.array([[x,y] for x in range(-n,n+1) for y in range(-n,n+1)])
            #batch = np.array([[x,y] for x in range(0,n+1) for y in range(0,n+1)])
            batch = np.expand_dims(np.expand_dims(batch, 1),1)
            vals = sess.run(target, {self.step_model.X:batch})
            #print(np.reshape(vals,[n+1,n+1]))
            #print(np.reshape(batch,[n+1,n+1,2])[:,:,1])
            vals = np.reshape(vals, [2*n+1,2*n+1,1])
            print(vals[n-3:n+4,n-3:n+3])

            # to test if flattening / reshaping working correctly
            #vals = np.array([x+y for x in range(-n,n+1) for y in range(-n,n+1)])
            vals = vals - np.amin(vals) # make only positive
            vals = vals / np.amax(vals) # make max elem 1
            vals = vals * 255 # scale to RGB
            vals = np.round(vals).astype(np.int8)
            #vals = np.reshape(vals, [n+1,n+1,1])
            im = np.repeat(vals, 3, axis=2)

            width = 6
            im = np.repeat(im, width, axis=0)
            im = np.repeat(im, width, axis=1)

            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(im)

            # let's also print out the value function:
            vals = self.step_model.value(batch)
            vals = np.reshape(vals, [2*n+1,2*n+1,1])
            #print("Values around root:")
            #print(vals[n-3:n+4,n-3:n+3])



        self.train = train
        self.viewer = None
        self.display = display
        #self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.train_only_progress = train_only_progress
        self.progress = step_model.progress_fun
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
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.int8)
        self.nc = nc
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.progress = np.array([0 for _ in range(nenv)])

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        #self.obs[:, :, :, -self.nc:] = obs
        self.obs[:, :, :, -self.nc:] = np.reshape(obs, (self.obs.shape))

    def run(self, rendering=False, display=False):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, = [],[],[],[],[]
        mb_progress_p, mb_progress_t, mb_real_rewards, = [],[],[] # stuff i've added
        mb_next_progress_p = []
        extra_progress_updates = []
        for n in range(self.nsteps):
            # get action, value of current state, progress pred of cur state.
            actions, values, progress_p = self.model.step(self.obs, self.dones)
            # save all observations, actions, values, dones, progress predictions
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_progress_p.append(progress_p)
            mb_progress_t.append(self.progress)
            # using actions, move in environment. Save rewards, obs, dones
            obs, rewards, env_dones, _ = self.env.step(actions)
            self.update_obs(obs)
            next_progress_p = self.model.progress(self.obs)
            mb_next_progress_p.append(next_progress_p)
            prog_dones = (next_progress_p < progress_p)
            for i,done in enumerate(prog_dones):
              if done:
                self.env.reset_i(i) # reset ith environment
                # save progress prediction and target for later training step
                extra_progress_updates.append((
                    self.obs[i],self.progress[i] + PROGRESS_MIN_STEP))
            self.dones = np.logical_or(env_dones, prog_dones)

            if rendering:
              self.env.render(1)
            if display:
              self.model.display('progress')
            mb_real_rewards.append(rewards)

            for n, done in enumerate(self.dones):
                if done:
                    # if env was just reset, zero all previous observations in stack.
                    self.obs[n,:,:,:-self.nc] = self.obs[n,:,:,:-self.nc]*0
                    # reset progress for env n
                    self.progress[n] = -PROGRESS_MIN_STEP
            mb_rewards.append(rewards)
            self.progress = self.progress + PROGRESS_MIN_STEP # udpate progress for each environment

        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.int8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_progress_p = np.asarray(mb_progress_p, dtype=np.float32).swapaxes(1, 0)
        mb_progress_t = np.asarray(mb_progress_t, dtype=np.float32).swapaxes(1, 0)
        mb_next_progress_p = np.asarray(mb_next_progress_p, dtype=np.float32).swapaxes(1, 0)
        mb_real_rewards = np.asarray(mb_real_rewards, dtype=np.int32).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.dones).tolist()

        
        # If the below is negative, then we are transitioning to 
        # TODO: should I use predicted or actual progress?
        # the below is always positive, unless last step (env resets when dec prog)
        progress_diffs = mb_next_progress_p - mb_progress_t  
        mb_progress_rewards = progress_diffs 
        mb_progress_rewards *= (1 - mb_dones) # zero intrinsic reward if done
        mb_progress_rewards *= PROGRESS_REWARD_SCALE


        mb_rewards = mb_rewards + mb_progress_rewards


        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
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
        mb_progress_p = mb_progress_p.flatten()
        mb_progress_t = mb_progress_t.flatten()
        mb_next_progress_p = mb_next_progress_p.flatten()
        return mb_obs, mb_rewards, mb_masks, mb_actions, mb_values, mb_real_rewards, mb_next_progress_p, mb_progress_rewards, mb_progress_t, extra_progress_updates

def learn(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=VF_COEF, ent_coef=MY_ENT_COEF, max_grad_norm=0.5, lr=LEARNING_RATE, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
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
    policy_loss_f = open(save_dir + "policy.txt","w", 1)
    entropy_loss_f = open(save_dir + "entropy.txt","w", 1)
    value_loss_f = open(save_dir + "value.txt","w", 1)
    total_loss_f = open(save_dir + "total.txt","w", 1)

    accumulated_rewards = 0
    for update in range(1, total_timesteps//nbatch+1):
        rendering = ((update // 30) % 50 == 0) and RENDERING
        display = (update % 200) == 1
        obs, rewards, masks, actions, values, real_rewards, next_progress, progress_rewards, progress_t, extra_progress_updates = runner.run(rendering, display)
        accumulated_rewards += np.sum(real_rewards) # only want to record times when we get positive reward
        progress_rewards = np.sum(progress_rewards)
        if np.sum(real_rewards > 0) > 0 and HALT_AFTER_REWARD:
          print("Found first reward after %d updates." % update)
          sys.exit()
        policy_loss, value_loss, policy_entropy, progress_loss, total_loss = model.train(obs, rewards, masks, actions, values, progress_t)
        progress_end_loss = model.train_only_progress(extra_progress_updates)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            progress_loss_f.write("%f\n" % progress_loss)
            policy_loss_f.write("%f\n" % policy_loss)
            entropy_loss_f.write("%f\n" % (- policy_entropy * MY_ENT_COEF))
            value_loss_f.write("%f\n" % value_loss)
            total_loss_f.write("%f\n" % (total_loss))
            ev = explained_variance(values, rewards)
            #logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("loss_total", float(total_loss))
            #logger.record_tabular("fps", fps)
            logger.record_tabular("progress_rewards", float(progress_rewards))
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("loss_policy_entropy", float(policy_entropy) * MY_ENT_COEF)
            logger.record_tabular("loss_value", float(value_loss) * VF_COEF)
            logger.record_tabular("loss_policy", float(policy_loss) * POLICY_LOSS_SCALE)
            logger.record_tabular("loss_progress", float(progress_loss))
            logger.record_tabular("loss_progress_end", (progress_end_loss))
            #logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("accumulated rewards", accumulated_rewards)
            logger.dump_tabular()
        save_interval = 4000
        if update % save_interval == 0:
            print("Saving model to %s" % save_path)
            model.save(save_path)
    env.close()
 

if __name__ == '__main__':
    main()
