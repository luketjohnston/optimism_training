import numpy as np
import tensorflow as tf
from baselines.terminate.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm, sample


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

            # First, the progress part of the model:
            progress_h1 = fc(tf.contrib.layers.flatten(X), 'progress_fc1', nh=16, init_scale=1.0/np.sqrt(16))
            progress_h2 = fc(progress_h1, 'progress_fc2', nh=16, init_scale=1.0/np.sqrt(16))
            progressf = fc(progress_h2, 'progress', 1, act=lambda x:x) # default act is relu

            # Now, the policy / value part:
            h1 = fc(tf.contrib.layers.flatten(X), 'fc1', nh=16, init_scale=1.0/np.sqrt(16))
            h2 = fc(h1, 'fc2', nh=16, init_scale=1.0/np.sqrt(16))
            h3 = fc(h2, 'fc3', nh=16, init_scale=1.0/np.sqrt(16))
            pi = fc(h3, 'pi', nact, act=lambda x:x) 
            vf = fc(h3, 'v', 1, act=lambda x:x)
            """ Outputs a number that we will require to be monotonically increasing
            throughout exploration of the environment."""


        v0 = vf[:, 0]
        progress0 = progressf[:, 0]
        a0 = sample(pi)

        # manual computation of desired progress:
        #progress0 = 0.1 * tf.reduce_sum(tf.abs(X), axis=[1,2,3])

        def step(ob, *_args, **_kwargs):
            a, v, o = sess.run([a0, v0, progress0], {X:ob})
            return a, v, o 

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        def progress(ob, *_args, **_kwargs):
            return sess.run(progress0,{X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
        self.progress = progress0
        self.progress_fun = progress
