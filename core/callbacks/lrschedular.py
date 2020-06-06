# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class CosineAnnealingScheduler(tf.keras.callbacks.Callback):
    def __init__(self,
                 learning_rate,
                 T_max,
                 eta_min=1e-6,
                 verbose=0
                 ):
        super(CosineAnnealingScheduler, self).__init__()
        self.eta_max = learning_rate
        self.T_max = T_max
        self.eta_min = eta_min

        self.global_step = 0
        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        learning_rate = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(learning_rate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            print('\nEpoch %s: current learning rate to %s.' % (epoch + 1, self.model.optimizer.learning_rate.numpy()))

    def on_batch_begin(self, batch, logs=None):
        learning_rate = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
                1 + np.cos(np.pi * (self.global_step % self.T_max) / self.T_max))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)


class WarmUpScheduler(tf.keras.callbacks.Callback):
    """warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate,
                 warmup_step,
                 global_step_init=0,
                 learning_rate_init=0.0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate {float} -- base learning rate.
        warmup_step {int} -- number of warmup steps. (default: {0})

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        learning_rate_init {float} -- initial learning rate for warm up. (default: {0.0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.warmup_step = warmup_step
        self.global_step = global_step_init
        self.learning_rate_init = learning_rate_init

        self.verbose = verbose
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        learning_rate = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        self.learning_rates.append(learning_rate)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            print('\nEpoch %s: current learning rate to %s.' % (epoch + 1, self.model.optimizer.learning_rate.numpy()))

    def on_batch_begin(self, batch, logs=None):
        slope = (self.learning_rate - self.learning_rate_init) / self.warmup_step
        warmup_rate = slope * self.global_step + self.learning_rate_init
        learning_rate = np.where(self.global_step < self.warmup_step, warmup_rate,
                                 self.learning_rate)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
