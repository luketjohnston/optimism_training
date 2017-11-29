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

from baselines.ordering.utils import discount_with_dones
from baselines.ordering.utils import Scheduler, make_path, find_trainable_variables
from baselines.ordering.policies import CnnPolicy
from baselines.ordering.utils import cat_entropy, mse


# the working params are 0.1 and 0.02, seed 2 and maybe 1?

# original default was 7e-4
LEARNING_RATE = 7e-4
# was working with -3 learning rate to learn order

ENV_N = 15

# NOTE: I have env setup to just move randomly. So
# no matter policy initialization, should have same order solution!

zero_all_except_order = False


ORDER_MIN_STEP = .001 # This seems to work well (.001)
#ORDER_MIN_STEP = 1.00 # This seems to work well (.001)

POLICY_LOSS_SCALE = 1000.0
#POLICY_LOSS_SCALE = 1.0
ORDER_REWARD_SCALE = 100.0
ORDER_REWARD_SCALE = 1.0
#ORDER_REG_SCALE = .001
ORDER_LOSS_SCALE = .1 / ORDER_MIN_STEP # this seems necessary for stability
#ORDER_LOSS_SCALE = 1.0
#ORDER_LOSS_SCALE = 1.0 

MY_ENT_COEF = 0.01 # originally 0.01
MY_ENT_COEF = 0.03 # originally 0.01
#MY_ENT_COEF = 0.00 # originally 0.01
HALT_AFTER_REWARD = False

VF_COEF = 0.5 # originally 0.5
VF_COEF = 500.0 # originally 0.5
#VF_COEF = 0.0

if zero_all_except_order:
  POLICY_LOSS_SCALE = 0.0
  ORDER_REWARD_SCALE = 0.0
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

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        PREV_ORDER = tf.placeholder(tf.float32, [nbatch]) # ordering output for previous timestep.

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model1 = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True) # model for previous step
        train_model2 = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True) # model for this step

        # the best solution to ordering will be for most visited states to have high order,
        # least visited states low order

        # penalize out of order things, but to prevent order from blowing up,
        # also have to penalize l2_loss of order magnitude.
        #order_loss = ORDER_LOSS_SCALE * tf.nn.relu(
        #    train_model.order - PREV_ORDER) + tf.nn.l2_loss(train_model.order)

        unscaled_order_loss = tf.reduce_mean(tf.sqrt(1e-4 + tf.nn.relu(ORDER_MIN_STEP - train_model2.order + train_model1.order)))
        unscaled_order_loss = tf.reduce_mean(tf.sqrt(1e-4 + tf.abs(ORDER_MIN_STEP - train_model2.order + train_model1.order)))
        #unscaled_order_loss = tf.reduce_mean(tf.abs(train_model2.order - train_model1.order - ORDER_MIN_STEP))
        scaled_order_loss = ORDER_LOSS_SCALE * unscaled_order_loss
        #regularization_order_loss = ORDER_REG_SCALE * tf.reduce_mean(tf.nn.l2_loss(train_model.order))
        regularization_order_loss = 0.0
        order_loss = (scaled_order_loss + regularization_order_loss)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model2.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model2.vf), R)) 
        entropy = tf.reduce_mean(cat_entropy(train_model2.pi)) 
        loss = pg_loss*POLICY_LOSS_SCALE - entropy*ent_coef + vf_loss * vf_coef + order_loss
        #loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, prev_obs):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model2.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr, 
                train_model1.X:prev_obs}
            if states != []:
                td_map[train_model2.S] = states
                td_map[train_model2.M] = masks
            policy_loss, value_loss, policy_entropy, order_loss0, total_loss, _ = sess.run(
                [pg_loss, vf_loss, entropy, order_loss, loss, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, order_loss0, total_loss

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

        def display(mode='order'):
            target = self.step_model.order if mode=='order' else self.step_model.value
            n = min(ENV_N, 20)
            #batch = np.array([[x,y] for x in range(-n,n+1) for y in range(-n,n+1)])
            batch = np.array([[x,y] for x in range(0,n+1) for y in range(0,n+1)])
            batch = np.expand_dims(np.expand_dims(batch, 1),1)
            vals = sess.run(target, {self.step_model.X:batch})
            #print(np.reshape(vals,[n+1,n+1]))
            #print(np.reshape(batch,[n+1,n+1,2])[:,:,1])

            # to test if flattening / reshaping working correctly
            #vals = np.array([x+y for x in range(-n,n+1) for y in range(-n,n+1)])
            vals = vals - np.amin(vals) # make only positive
            vals = vals / np.amax(vals) # make max elem 1
            vals = vals * 255 # scale to RGB
            vals = np.round(vals).astype(np.int8)
            #vals = np.reshape(vals, [2*n+1,2*n+1,1])
            vals = np.reshape(vals, [n+1,n+1,1])
            im = np.repeat(vals, 3, axis=2)

            width = 6
            im = np.repeat(im, width, axis=0)
            im = np.repeat(im, width, axis=1)

            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(im)

        self.train = train
        self.viewer = None
        self.display = display
        #self.train_model = train_model
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
        self.prev_obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.prev_raw_obs = obs # reset environment, and save for prev raw obs
        self.update_obs(obs)
        self.update_prev_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.last_order = [0 for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        #self.obs[:, :, :, -self.nc:] = obs
        self.obs[:, :, :, -self.nc:] = np.reshape(obs, (self.obs.shape))

    def update_prev_obs(self, obs):
        self.prev_obs = np.roll(self.prev_obs, shift=-self.nc, axis=3)
        self.prev_obs[:, :, :, -self.nc:] = np.reshape(obs, (self.prev_obs.shape))

    def run(self, rendering=False, display=False):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, = [],[],[],[],[]
        mb_order, mb_last_order, mb_real_rewards, mb_prev_obs = [],[],[],[] # stuff i've added
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, order = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_prev_obs.append(np.copy(self.prev_obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_order.append(order)
            mb_last_order.append(self.last_order) # for minibatch, need to know previous order


            self.last_order = order # save order from this step for use with next step
            obs, rewards, dones, _ = self.env.step(actions)
            if rendering:
              self.env.render(1)
            if display:
              self.model.display('order')
            mb_real_rewards.append(rewards)
            # save order to be formatted for minibatch later
            #mb_order_t.append(self.order) 
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    self.last_order[n] = 0 # order starts at 0 for new env
            self.update_obs(obs)
            self.update_prev_obs(self.prev_raw_obs)
            self.prev_raw_obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_prev_obs = np.asarray(mb_prev_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_order = np.asarray(mb_order, dtype=np.float32).swapaxes(1, 0)
        mb_last_order = np.asarray(mb_last_order, dtype=np.float32).swapaxes(1, 0)
        #mb_order_t = np.asarray(mb_order_t, dtype=np.int32).swapaxes(1, 0)
        mb_real_rewards = np.asarray(mb_real_rewards, dtype=np.int32).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # compute order reward. Can only be negative.
        # if we run with the assumption that "order" will actually converge to estimate
        # of frequency of reaching state, then we want to go places that are seldom visited:

        order_diffs = mb_order - mb_last_order # want this to be negative
        mb_order_rewards = (order_diffs) # reward increasing order, penalize decreasing it.
        #mb_order_rewards = -1 * order_diffs
        # actually we need the sqrt, it encourages long loops... I think? otherwises the discount is all that is encouraging long loops.
        #mb_order_rewards = (-1 * np.sqrt(np.maximum(ORDER_MIN_STEP + order_diffs, 0.0))) # remove sqrt here, don't want loops to be +EV
        #mb_order_rewards = (-1 * np.abs(ORDER_MIN_STEP - order_diffs))
        # update rewards with order rewards.
        if mb_actions[0,0] == 2:
          #print(mb_order_rewards[0,0])
          pass
        mb_rewards = mb_rewards + ORDER_REWARD_SCALE * mb_order_rewards

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
        mb_order = mb_order.flatten()
        mb_last_order = mb_last_order.flatten()
        return mb_obs, mb_prev_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_real_rewards, mb_last_order, mb_order_rewards

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
    order_loss_f = open(save_dir + "order.txt", "w", 1) # 1 is line buffered
    policy_loss_f = open(save_dir + "policy.txt","w", 1)
    entropy_loss_f = open(save_dir + "entropy.txt","w", 1)
    value_loss_f = open(save_dir + "value.txt","w", 1)
    total_loss_f = open(save_dir + "total.txt","w", 1)

    accumulated_rewards = 0
    for update in range(1, total_timesteps//nbatch+1):
        rendering = ((update // 30) % 50 == 0) and RENDERING
        display = (update % 200) == 1
        obs, prev_obs, states, rewards, masks, actions, values, real_rewards, last_order, order_rewards = runner.run(rendering, display)
        accumulated_rewards += np.sum(real_rewards) # only want to record times when we get positive reward
        order_rewards = np.sum(order_rewards)
        if np.sum(real_rewards > 0) > 0 and HALT_AFTER_REWARD:
          print("Found first reward after %d updates." % update)
          sys.exit()
        policy_loss, value_loss, policy_entropy, order_loss, total_loss = model.train(obs, states, rewards, masks, actions, values, prev_obs)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            order_loss_f.write("%f\n" % order_loss)
            policy_loss_f.write("%f\n" % policy_loss)
            entropy_loss_f.write("%f\n" % (policy_entropy * MY_ENT_COEF))
            value_loss_f.write("%f\n" % value_loss)
            total_loss_f.write("%f\n" % (total_loss))
            ev = explained_variance(values, rewards)
            #logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("loss_total", float(total_loss))
            #logger.record_tabular("fps", fps)
            logger.record_tabular("order_rewards", float(order_rewards))
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("loss_policy_entropy", float(policy_entropy) * MY_ENT_COEF)
            logger.record_tabular("loss_value", float(value_loss) * VF_COEF)
            logger.record_tabular("loss_policy", float(policy_loss) * POLICY_LOSS_SCALE)
            logger.record_tabular("loss_order", float(order_loss))
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
