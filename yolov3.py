# -*- coding: utf-8 -*-
import tensorflow as tf
slim  = tf.contrib.slim

import config

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
                      weights_regularizer=slim.l2_regularizer(weight_decay),# can be removed
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params
                      ) as arg_sc:
    return arg_sc

def backbone(inputs, n_classes, is_training=True, scope='yolov3'):
  """Yolov3 backbone
  """
  img_size = inputs.get_shape().as_list()[1:3][::-1]

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
      yolo_1 = yolo_head(net, config.anchor[6:9], n_classes, img_size, scope='yolo_1')

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
      yolo_2 = yolo_head(net, config.anchor[3:6], n_classes, img_size, scope='yolo_2')

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
      yolo_3 = yolo_head(net, config.anchor[0:3], n_classes, img_size, scope='yolo_3')

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
  Params:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The growth rate of the dense layer.
    scope: Optional variable_scope.
  Return:
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

def yolo_head(inputs, anchors, n_classes, img_size, scope='yolo_head'):
  """Head layer
  Convert output of yolo network to bounding box parameters

  Params:
    img_size: w, h

  Return:
    xc, yc, w, h, confidence, label
  """
  with tf.name_scope(scope, 'yolo_head', [inputs]):
    n_anchors = len(anchors)
    bbox_attrs = 5+n_classes

    predictions = inputs
    grid_size = predictions.get_shape().as_list()[1:3][::-1]
    n_dims = grid_size[0] * grid_size[1]
    predictions = tf.reshape(predictions, [-1, n_anchors * n_dims, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(predictions,
                                                           [2, 2, 1, n_classes],
                                                           axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)

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


def loss(predictions, labels, n_classes, ignore_thresh=0.5, scope='yolo_loss'):
  """Yolo loss
  """
  lambda_coord = 5.0
  lambda_noobj = 0.5

  loss = 0.0

  assert len(predictions)==len(labels), "prediction and labels should have the same layers"
  batch_size = predictions.shape[0]

  with tf.name_scope(scope, 'yolo_loss', [predictions, labels]):

    bbox, conf, classes = tf.split(predictions,
                                    [4,1,n_classes],
                                    axis=-1)
    bbox_hat,conf_hat,classes_hat = tf.split(labels,
                                             [4,1,n_classes],
                                             axis=-1)

    # Find ignore mask, iterate over each of batch.
    ignore_mask = tf.TensorArray(labels.dtype, size=1, dynamic_size=True)
    object_mask_bool = tf.cast(conf_hat, 'bool')

    def loop_body(b, ignore_mask):
      true_box = tf.boolean_mask(bbox_hat[b], object_mask_bool[b,...,0])
      iou = box_iou(bbox[b], true_box)
      best_iou = tf.maximum(iou, axis=-1)
      ignore_mask = ignore_mask.write(b, tf.cast(best_iou<ignore_thresh, true_box.dtype))
      return b+1, ignore_mask

    _, ignore_mask = tf.s.while_loop(lambda b,*args: b<batch_size, loop_body, [0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    obj = conf_hat
    noobj = 1. - obj

    coord_loss = obj*(tf.reduce_sum(tf.square(bbox[..., 0:2]-bbox_hat[..., 0:2]), keepdims=True)+
                      tf.reduce_sum(tf.square(tf.sqrt(bbox[..., 2:4])-tf.sqrt(bbox_hat[..., 2:4])), keepdims=True))
    coord_loss *= lambda_coord
    coord_loss = tf.reduce_sum(coord_loss,axis=1)

    iou_loss = tf.reduce_sum(obj*tf.nn.sigmoid_cross_entropy_with_logits(logits=conf, labels=conf_hat) +
                             lambda_noobj*noobj*tf.nn.sigmoid_cross_entropy_with_logits(logits=conf, labels=conf_hat)
                             *ignore_mask,
                             axis=1)
    class_loss = tf.reduce_sum(obj*tf.nn.sigmoid_cross_entropy_with_logits(logits=classes, labels=classes_hat),
                               axis=1)

    _loss = coord_loss + iou_loss + class_loss
    loss += tf.reduce_mean(_loss)

    tf.losses.add_loss(loss)
    loss = tf.losses.get_total_loss()

  return loss

def box_iou(box1, box2):
  """Computes Intersection over Union value for 2 bounding boxes

  Params:
    xc, yc, w, h
  """
  box1_xyxy = yolo_boxes_to_corners(box1)
  box2_xyxy = yolo_boxes_to_corners(box2)

  intersect_min = tf.maximum(box1_xyxy[...,0:2], box1_xyxy[...,0:2])
  intersect_max = tf.minimum(box2_xyxy[...,2:4], box2_xyxy[...,2:4])
  intersect_wh = tf.maximum(intersect_max-intersect_min, 0)
  intersect_area = intersect_wh[..., 0]*intersect_wh[..., 1]

  bbox1_area = box1[..., 2]*box1[..., 3]
  bbox2_area = box2[..., 2]*box2[..., 3]

  eps = 1e-5
  iou = intersect_area/(bbox1_area+bbox2_area-intersect_area+eps)
  return iou

def yolo_boxes_to_corners(bbox):
  """Convert xc, yc, w, h to xmin, ymin, xmax, ymax along last axis
  """
  bbox_others = tf.split(bbox, [4, -1], axis=-1)
  bbox_min = bbox[...,0:2] - bbox[...,2:4]/2.
  bbox_max = bbox[...,0:2] + bbox[...,2:4]/2.
  bbox = tf.concat([bbox_min, bbox_max, bbox_others], axis=-1)
  return bbox





