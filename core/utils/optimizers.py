# -*- coding: utf-8 -*-
#################################################################
# Code from https://github.com/LJNL/accum_optimizer_for_keras
#################################################################
import tensorflow as tf


class Accumulative(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, accum_steps=1, name='accum', **kwargs):
        self.name = name
        super(Accumulative, self).__init__(name, **kwargs)
        self.optimizer = optimizer
        with tf.name_scope(self.__class__.__name__):
            self.accum_steps = accum_steps
            self.iterations = tf.Variable(0, dtype='int64', name='iterations')
            self.cond = tf.equal(self.iterations % self.accum_steps, 0)
            self.lr = self.optimizer.lr
            self.optimizer.lr = tf.cond(self.cond, lambda: self.optimizer.lr.value(), lambda: 0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, tf.cond(self.cond, lambda: value.value(), lambda: 1 - 1e-7))
            for attr in self.optimizer.get_config():
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)

            self._create_slots = self.optimizer._create_slots
            self._resource_apply_dense = self.optimizer._resource_apply_dense

            def get_gradients(loss, params):
                return [ag / self.accum_steps for ag in self.accum_grads]

            self.optimizer.get_gradients = get_gradients

    def get_updates(self, loss, params):
        self.iterations = tf.add(self.iterations, 1)
        self.optimizer.iterations = tf.add(self.optimizer.iterations, tf.cast(self.cond, 'int64'))
        self.updates = [
            self.iterations,
            self.optimizer.iterations
        ]
        # gradient accumulation
        self.accum_grads = [tf.zeros(p.shape, dtype=p.dtype) for p in params]
        grads = self.get_gradients(loss, params)

        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(ag=tf.cond(self.cond, lambda: g, lambda: ag + g))

        # inheriting updates of original optimizer
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        return self.updates

    def get_config(self):
        iterations = self.iterations.numpy()
        self.iterations = 0
        config = self.optimizer.get_config()
        self.iterations = iterations
        return config
