import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample


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
            """ The part of the model that predicts the progress through the 
            environment we don't actually have to predict the number of frames. 
            Could just have a monotonically increasing prediction. This is 
            enough to enforce that the model learns an order to the states that 
            corresponds to progress."""
            progressf = fc(h4, 'progress', 1, act=lambda x:x) # default act is relu

        v0 = vf[:, 0]
        progress0 = progressf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v, p = sess.run([a0, v0, progress0], {X:ob})
            return a, v, [], p #[] is a dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.progress = progress0
