# -*- coding: utf-8 -*-
import tensorflow as tf
slim  = tf.contrib.slim

_ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]

def arg_scope(batch_norm_decay=0.997,
              batch_norm_epsilon=1e-5,
              batch_norm_scale=True,
              is_training=True,
              weight_decay=1e-4):

  leaky_relu = lambda x: tf.nn.leaky_relu(x, 0.1)
  batch_norm_params = {'decay': batch_norm_decay,
                       'epsilon': batch_norm_epsilon,
                       'scale': batch_norm_scale,
                       'is_training': is_training
                       }
  with slim.arg_scope([slim.conv2d],
                      activation_fn=leaky_relu,
                      weights_initializer=slim.variance_scaling_initializer(),
                      weights_regularizer=slim.l2_regularizer(weight_decay),# can be move
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params
                      ) as arg_sc:
    return arg_sc

def backbone(inputs, n_classes, is_training=True, scope='yolov3'):
  """Yolov3 backbone
  """
  img_size = inputs.get_shape().as_list()[1:3]

  with slim.arg_scope(arg_scope(is_training=is_training)):
    with tf.variable_scope(scope, 'yolov3', [inputs]):
      # 0
      net = slim.conv2d(inputs, 32, 3)
      # 1-4
      net = conv2d_same(net, 64, 3, 2, scope='conv_same_1')
      net = residual(net, 64, 1, scope='residual_1')
      # 5-11
      net = conv2d_same(net, 128, 3, 2, scope='conv_same_2')
      net = residual(net, 128, 2, scope='residual_2')
      # 12-36
      net = conv2d_same(net, 256, 3, 2, scope='conv_same_3')
      net = residual(net, 256, 8, scope='residual_3')
      route_36 = net
      # 37-61
      net = conv2d_same(net, 512, 3, 2, scope='conv_same_4')
      net = residual(net, 512, 8, scope='residual_4')
      route_61 = net
      # 62-74
      net = conv2d_same(net, 1024, 3, 2, scope='conv_same_5')
      net = residual(net, 1024, 4, scope='residual_5')
      # 75-78
      net = slim.conv2d(net, 512, 1)
      net = slim.conv2d(net, 1024, 3)
      net = slim.conv2d(net, 512, 1)
      net = slim.conv2d(net, 1024, 3)

      # 79-81
      net = slim.conv2d(net, 512, 1)
      route_79 = net
      net = slim.conv2d(net, 1024, 3)
      net = slim.conv2d(net, 255, 1,
                        activation_fn=None, normalizer_fn=None)
      # 82, yolo
      yolo_1 = detect(net, _ANCHORS[6:9], n_classes, img_size, scope='yolo_1')

      # 83, route 79
      net = tf.concat([route_79], axis=-1)
      # 84
      net = slim.conv2d(net, 256, 1)
      # 85
      net = upsample(net, 256, scope='upsample_1')

      # 86, route 85 61
      net = tf.concat([net, route_61], axis=-1)
      # 87-90
      net = slim.conv2d(net, 256, 1)
      net = slim.conv2d(net, 512, 3)
      net = slim.conv2d(net, 256, 1)
      net = slim.conv2d(net, 512, 3)

      # 91
      net = slim.conv2d(net, 256, 1)
      route_91 = net

      # 92-93
      net = slim.conv2d(net, 512, 3)
      net = slim.conv2d(net, 255, 1,
                        activation_fn=None, normalizer_fn=None)
      # 94, yolo
      yolo_2 = detect(net, _ANCHORS[3:6], n_classes, img_size, scope='yolo_2')


      # 95, route
      net = tf.concat([route_91], axis=-1)
      # 96
      net = slim.conv2d(net, 128, 1)
      # 97
      net = upsample(net, 128, scope='upsample_2')

      # 98, route 97 36
      net = tf.concat([net, route_36], axis=-1)
      # 99-105
      net = slim.conv2d(net, 128, 1)
      net = slim.conv2d(net, 256, 3)
      net = slim.conv2d(net, 128, 1)
      net = slim.conv2d(net, 256, 3)
      net = slim.conv2d(net, 128, 1)
      net = slim.conv2d(net, 256, 3)
      net = slim.conv2d(net, 255, 1,
                        activation_fn=None, normalizer_fn=None)
      # yolo
      yolo_3 = detect(net, _ANCHORS[0:3], n_classes, img_size, scope='yolo_3')

  detections = tf.concat([yolo_1, yolo_2, yolo_3], axis=1)
  return detections

def residual(inputs, depth, n_unit, scope='residual'):
  net = inputs
  with tf.name_scope(scope, 'residual', [net]):
    for i in range(n_unit):
      identity = net
      output= unit(net, depth, scope='unit_%d' %(i+1))
      net = tf.add(identity, output, name='shortcut_%d' %(i+1))

  return net

def unit(inputs, depth, scope='unit'):
  """Unit.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The growth rate of the dense layer.
    scope: Optional variable_scope.
  Returns:
    The dense layer's output.
  """

  with tf.name_scope(scope, 'unit', [inputs]):
    net = slim.conv2d(inputs, depth//2, 1)
    net = slim.conv2d(net, depth, 3)

  return net


def upsample(inputs, depth, method='upsample_bilinear', scope='upsmaple'):
  """Upsample
  """
  with tf.variable_scope(scope, 'upsample', [inputs]):
    if method == 'conv2d_transpose':
      net = slim.conv2d_transpose(inputs, depth, 3, stride=2,
                                  activation_fn=None, biases_initializer=None)
    elif method == 'upsample_bilinear':
      height, width = inputs.get_shape().as_list()[1:3]
      net = tf.image.resize_nearest_neighbor(inputs, [2*height, 2*width])

    return net

def conv2d_same(inputs,
                num_outputs,
                kernel_size,
                stride,
                rate=1,
                scope='conv_same'):
  """customized conv2d with 'SAME' padding
  """
  with tf.name_scope(scope, 'conv_same', [inputs]):
    if stride==1:
      return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                         padding='SAME')
    else:
      kernel_size_effective = kernel_size+(kernel_size-1)*(rate-1)
      pad_total = kernel_size_effective-1
      pad_beg = pad_total//2
      pad_end = pad_total-pad_beg
      inputs = tf.pad(inputs,
                      [[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                         rate=rate, padding='VALID')

def detect(inputs, anchors, n_classes, img_size, scope='detection'):
  """Detect layer
  """
  with tf.name_scope(scope, 'detection',[inputs]):
    n_anchors = len(anchors)
    bbox_attrs = 5+n_classes

#    predictions = slim.conv2d(inputs, n_anchors*bbox_attrs, 1, stride=1,
#                              activation_fn=None, normalizer_fn=None)
    predictions = inputs
    grid_size = predictions.get_shape().as_list()[1:3]
    n_dims = grid_size[0] * grid_size[1]
    predictions = tf.reshape(predictions, [-1, n_anchors * n_dims, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(predictions,
                                                           [2, 2, 1, n_classes],
                                                           axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_y = tf.range(grid_size[0], dtype=tf.float32)
    grid_x = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, n_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [n_dims, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)

    return predictions