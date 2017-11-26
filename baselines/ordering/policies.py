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
        X = tf.placeholder(tf.uint8, shape=[nbatch,nh,nw,nc*nstack]) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=1, stride=1, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=1, stride=1, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=1, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x) 
            vf = fc(h4, 'v', 1, act=lambda x:x)
            """ Outputs a number that we will require to be monotonically increasing
            throughout exploration of the environment."""
            orderf = fc(h4, 'progress', 1, act=lambda x:x) # default act is relu

        v0 = vf[:, 0]
        order0 = orderf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v, o = sess.run([a0, v0, order0], {X:ob})
            return a, v, [], o #[] is a dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.order = order0
