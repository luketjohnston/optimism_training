import numpy as np
import tensorflow as tf
from baselines.ordering.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        #nh, nw, nc = ob_space.shape
        nh,nw,nc = 1,1,2
        ob_shape = (nbatch, nh, nw, nc*nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, shape=[None,nh,nw,nc*nstack]) #obs
        with tf.variable_scope("model", reuse=reuse):
            # For now, let's use a fully connected model:

            # First, the order part of the model:
            order_h1 = fc(tf.contrib.layers.flatten(X), 'order_fc1', nh=16, init_scale=np.sqrt(2))
            order_h2 = fc(order_h1, 'order_fc2', nh=16, init_scale=np.sqrt(2))
            orderf = fc(order_h2, 'order', 1, act=lambda x:x) # default act is relu

            # Now, the policy / value part:
            h1 = fc(tf.contrib.layers.flatten(X), 'fc1', nh=16, init_scale=np.sqrt(2))
            h2 = fc(tf.contrib.layers.flatten(X), 'fc2', nh=16, init_scale=np.sqrt(2))
            pi = fc(h2, 'pi', nact, act=lambda x:x) 
            vf = fc(h2, 'v', 1, act=lambda x:x)
            """ Outputs a number that we will require to be monotonically increasing
            throughout exploration of the environment."""


        v0 = vf[:, 0]
        order0 = orderf[:, 0]
        #order0 = tf.reduce_sum(X, axis=[1,2,3]) * 1
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        # manual computation of desired order:
        # order0 = tf.reduce_sum(X, axis=[1,2,3])

        def step(ob, *_args, **_kwargs):
            a, v, o = sess.run([a0, v0, order0], {X:ob})
            return a, v, [], o #[] is a dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        def order(ob, *_args, **_kwargs):
            return sess.run(order0,{X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.order = order0
        self.order_fun = order
