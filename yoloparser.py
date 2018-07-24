# -*- coding: utf-8 -*-
from collections import OrderedDict

import numpy as np
import tensorflow as tf

class YoloParser:

  def __init__(self, cfg_path, weights_path, output_path, input_dims=3):
    self._input_dims = input_dims
    self._cfg_path = cfg_path
    self._weights_path = weights_path
    self._output_path = output_path

  def run(self):
    """One shoot one kill
    """
    print('Reading .cfg file ...')
    cfg = self.read_file(self._cfg_path)
    print('Converting ...')
    print('From %s'%self._weights_path)
    print('To   %s'%self._output_path)
    weights = self.decode(cfg, self._weights_path)
    self.encode(weights, self._output_path)
    print('Finish !')

  def read_file(self, path, split='='):
    cfg = OrderedDict()
    nlayer = -1
    with open(path) as file:
      for line in file:
        line = line.strip()
        if not len(line) or line.startswith('#'):
          continue

        if line.startswith('['):
          nlayer += 1
          section = line.strip('[]')
          cfg[nlayer] = {'section': section}
        else:
          key, value = line.split(split)
          cfg[nlayer][key.strip()] = value.strip()
    return cfg

  def decode(self, cfg, path, split=','):

    weights_file = open(path, 'rb')

    # Just read them
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    prev_filter = self._input_dims
    n_conv = 0
    for nlayer in cfg:
      if cfg[nlayer]['section'].startswith('convolutional'):
        filters = int(cfg[nlayer]['filters'])
        size = int(cfg[nlayer]['size'])
        batch_normalize = 'batch_normalize' in cfg[nlayer]

        # Setting weights.
        # Darknet serializes convolutional weights as:
        # [bias/beta, [gamma, mean, variance], conv_weights]

        weights_shape = (size, size, prev_filter, filters)
        darknet_w_shape = (filters, weights_shape[2], size, size)
        weights_size = np.product(weights_shape)

        conv_bias = np.ndarray(
            shape=(filters, ),
            dtype='float32',
            buffer=weights_file.read(filters * 4))

        if batch_normalize:
          bn_weights = np.ndarray(
              shape=(3, filters),
              dtype='float32',
              buffer=weights_file.read(filters * 12))
        else:
          bn_weights = None

        conv_weights = np.ndarray(
            shape=darknet_w_shape,
            dtype='float32',
            buffer=weights_file.read(weights_size * 4))

        prev_filter = filters
        #print('Convolution %d:'%n_conv, cfg[nlayer]['section'], conv_weights.shape)
        yield 'Conv_%d'%n_conv, conv_weights, conv_bias, bn_weights
        n_conv += 1

      elif cfg[nlayer]['section'].startswith('route'):
        prev_filter = 0
        for i in cfg[nlayer]['layers'].split(split):
          i = int(i.strip())
          if i < 0:
            i += nlayer

          while not cfg[i]['section'].startswith('convolutional'):
            i -= 1

          prev_filter += int(cfg[i]['filters'])
      elif cfg[nlayer]['section'].startswith('shortcut'):
        prev_filter = int(cfg[nlayer-1]['filters'])

      elif cfg[nlayer]['section'].startswith('upsample'):
        prev_filter = int(cfg[nlayer-1]['filters'])

      elif cfg[nlayer]['section'].startswith('yolo'):
        prev_filter = int(cfg[nlayer-1]['filters'])

      elif cfg[nlayer]['section'].startswith('maxpool'):
        prev_filter = int(cfg[nlayer-1]['filters'])

      elif cfg[nlayer]['section'].startswith('net'):
          pass
      else:
          raise ValueError(
              'Unsupported section header type: {}'.format(cfg[nlayer]['section']))

    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()

    if remaining_weights > 0:
      print('Warning: unused weights {}'.format(remaining_weights))
    else:
      print('Success!')

  def encode(self, weights, output_path):
    """ weights: (section, conv_weights, conv_bias/beta, bn_weights)
                  bn_weights: (gamma, mean, variance)
        output_path:
        name_kw: {default_conv_name: custom_conv_name}
    """
    print('Encode weights...')
    var_list = []
    for section, conv_weights, conv_bias, bn_weights in weights:
      # DarkNet conv_weights are serialized Caffe-style:
      # (out_dim, in_dim, height, width)
      # We would like to set these to Tensorflow order:
      # (height, width, in_dim, out_dim)
      conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
      # encode
      with tf.variable_scope(section) :
        weights = tf.Variable(conv_weights, name='weights')
        var_list.append(weights)

        if bn_weights is None:
          var_list.append(tf.Variable(conv_bias, name='biases'))
        else:
          bn_gamma = bn_weights[0]
          bn_beta = conv_bias
          bn_mean = bn_weights[1]
          bn_var = bn_weights[2]

          with tf.variable_scope('BatchNorm'):
            var_list.append(tf.Variable(bn_gamma, name='gamma'))
            var_list.append(tf.Variable(bn_beta, name='beta'))
            var_list.append(tf.Variable(bn_mean, name='moving_mean'))
            var_list.append(tf.Variable(bn_var, name='moving_variance'))

    print('Model Parameters:')
    for var in var_list:
      print(var)

    saver = tf.train.Saver(var_list)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      saver.save(sess, output_path)