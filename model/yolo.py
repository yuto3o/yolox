# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
slim  = tf.contrib.slim

##########################
#        BACKBONE        #
##########################
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
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params
                      ) as arg_sc:
    return arg_sc

def backbone(inputs, n_classes, num_anchors_per_layer, is_training=True, scope='yolov3'):
  """Yolov3 backbone
  """

  with slim.arg_scope(arg_scope(is_training=is_training)):
    with tf.variable_scope(scope, 'yolov3', [inputs], reuse=tf.AUTO_REUSE):
      # 0
      net = slim.conv2d(inputs, 32, 3, )
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
      net = slim.conv2d(net, num_anchors_per_layer*(n_classes+5), 1,
                        activation_fn=None, normalizer_fn=None)
      # 82, yolo
      yolo_1 = net

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
      net = slim.conv2d(net, num_anchors_per_layer*(n_classes+5), 1,
                        activation_fn=None, normalizer_fn=None)
      # 94, yolo
      yolo_2 = net

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
      net = slim.conv2d(net, num_anchors_per_layer*(n_classes+5), 1,
                        activation_fn=None, normalizer_fn=None)
      # yolo
      yolo_3 = net

      logits = [yolo_3, yolo_2, yolo_1]
  return logits

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

##########################
#        WRAPPER         #
##########################

class YOLOv3(object):

  def __init__(self, anchors, num_classes, **kwargs):
    self.ANCHORS = np.array(anchors)
    self.NUM_PER_LAYER = 3
    self.NUM_CLASSES = num_classes
    self.ANCHOR_MASK = [[0,1,2],[3,4,5],[6,7,8]]

    self.IOU_THRESHOLD = kwargs.get('iou_threshold', 0.5)
    self.SCORE_THRESHOLD = kwargs.get('score_threshold', 0.5)
    self.MAX_OUTPUT_BOXES = kwargs.get('max_output_boxes', 8)

    self.IMG_DIM = kwargs.get('img_dim', 3)

  def load_weight():
    restore_dict = {}
    var_list = slim.get_model_variables()
    for var in var_list[:5]:
      restore_dict['Conv_0/'+var.name.split('/', 2)[-1].split(':')[0]] = var
    for var in var_list[5:]:
      restore_dict[var.name.split('/', 1)[-1].split(':')[0]] = var
    saver = tf.train.Saver(restore_dict)

    return saver

  def load_coco_name(path):
    coco = {}
    with open(path, 'rt') as file:
      for index, label in enumerate(file):
        coco[index] = label.strip()

    return coco

  def decode_feature(self, feats):

    mult_feature = []

    for feat in feats:
      grid_size = feat.get_shape().as_list()[1:3][::-1]
      n_dims = grid_size[0] * grid_size[1]
      n_anchors = self.NUM_PER_LAYER
      bbox_attrs = self.NUM_CLASSES+5

      feat = tf.reshape(feat, [-1, n_anchors * n_dims, bbox_attrs])
      mult_feature.append(feat)

    feats = tf.concat(mult_feature, axis=1)

    xcyc, wh, conf, cls = tf.split(feats, [2,2,1,self.NUM_CLASSES], axis=-1)

    def nms(boxes, scores, max_output_size, iou_thresh, score_thresh):
      def _nms(boxes, scores):
        idx = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_thresh, score_thresh)
        boxes_nms = tf.gather(boxes, idx)
        scores_nms = tf.gather(scores, idx)
        padding = tf.maximum(max_output_size-tf.shape(boxes_nms)[0], 0)

        boxes_nms = tf.pad(boxes_nms, [(0, padding), (0, 0)])
        scores_nms = tf.pad(scores_nms, [(0, padding)])

        return boxes_nms, scores_nms

      outputs = tf.map_fn(fn=lambda x: _nms(x[0], x[1]),
                          elems=[boxes, scores],
                          dtype=(tf.float32, tf.float32))
      return outputs

    def yolo_bbox_to_corners(xcyc, wh):
      box_mins = xcyc - (wh / 2.)
      box_maxs = xcyc + (wh / 2.)
      box = tf.concat([box_mins, box_maxs], axis=-1)

      return box

    boxes = yolo_bbox_to_corners(xcyc, wh)
    scores = conf*cls

    boxes_out = []
    scores_out = []
    labels_out = []
    for cls in range(self.NUM_CLASSES):
      _boxes, _scores = nms(boxes, scores[..., cls],
                            max_output_size=self.MAX_OUTPUT_BOXES,
                            iou_thresh=self.IOU_THRESHOLD,
                            score_thresh=self.SCORE_THRESHOLD)
      _label = tf.ones_like(_scores, dtype=tf.float32) * tf.cast(cls, dtype=tf.float32)

      boxes_out.append(_boxes)
      scores_out.append(_scores)
      labels_out.append(_label)

    boxes_out = tf.concat(boxes_out, axis=1)
    scores_out = tf.concat(scores_out, axis=1)
    labels_out = tf.concat(labels_out, axis=1)

    return boxes_out, scores_out, labels_out


  def forward(self, x, y=None, is_training=False):

    img_h, img_w = x.get_shape().as_list()[1:3]
    logits = backbone(x, self.NUM_CLASSES, self.NUM_PER_LAYER, is_training)

    outputs = []
    losses = 0.

    for i, logit in enumerate(logits):
      num_batch = tf.cast(tf.shape(logit)[0], tf.float32)

      anchors = self.ANCHORS[self.ANCHOR_MASK[i]]
      grid_h, grid_w = logit.get_shape().as_list()[1:3]
      logit = tf.reshape(logit, [-1, grid_h, grid_w, self.NUM_PER_LAYER, self.NUM_CLASSES + 5])

      anchors = tf.cast(tf.reshape(anchors,[1, 1, 1, self.NUM_PER_LAYER, 2]), tf.float32)

      grid_x, grid_y = tf.meshgrid(tf.range(grid_w, dtype=tf.float32),
                                 tf.range(grid_h, dtype=tf.float32))
      grid_x = tf.expand_dims(grid_x, axis=-1)
      grid_y = tf.expand_dims(grid_y, axis=-1)
      grid = tf.concat([grid_x, grid_y], axis=-1)
      grid = tf.expand_dims(grid, axis=-2)

      pred_xcyc = tf.nn.sigmoid(logit[..., :2]) + grid
      pred_wh = tf.exp(logit[..., 2:4]) * anchors
      pred_conf = tf.nn.sigmoid(logit[..., 4:5])
      pred_cls = logit[..., 5:]

      if is_training:
        true_xcyc = y[i][..., 0:2]
        true_wh = y[i][..., 2:4]
        true_conf = y[i][..., 4:5]
        true_cls = y[i][..., 5:]

        ###############
        # ignore mask #
        ###############
        true_boxes = tf.py_func(self.pick_out_gt_box, [y[i]], [tf.float32] )[0]

        true_boxes_xy = true_boxes[..., 0:2]
        true_boxes_wh = true_boxes[..., 2:4]
        ## for broadcasting
        logits_xy = tf.expand_dims(pred_xcyc, axis=-2)
        logits_wh = tf.expand_dims(pred_wh, axis=-2)

        logits_wh_half = logits_wh / 2.
        logits_min = logits_xy - logits_wh_half
        logits_max = logits_xy + logits_wh_half

        true_boxes_wh_half =  true_boxes_wh/ 2.
        true_boxes_min = true_boxes_xy - true_boxes_wh_half
        true_boxes_max = true_boxes_xy + true_boxes_wh_half

        intersect_min = tf.maximum(logits_min, true_boxes_min)
        intersect_max = tf.minimum(logits_max, true_boxes_max)

        intersect_wh = tf.maximum(intersect_max-intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        logits_area = logits_wh[..., 0] * logits_wh[..., 1]
        true_boxes_area = true_boxes_wh[..., 0] * true_boxes_wh[..., 1]

        union_area = logits_area + true_boxes_area - intersect_area
        iou_scores = intersect_area / union_area

        best_ious = tf.reduce_max(iou_scores, axis=-1, keepdims=True)
        ignore_mask = tf.cast(best_ious < self.IOU_THRESHOLD, tf.float32)

        obj_mask = true_conf
        box_scale = tf.exp(true_wh) * tf.cast(anchors, tf.float32)
        box_scale = 2 - box_scale[..., 0:1] * box_scale[..., 1:2]

        xy_delta    = obj_mask     * (pred_xcyc-true_xcyc)     * box_scale
        wh_delta    = obj_mask     * (pred_wh-true_wh)     * box_scale
        conf_delta  = obj_mask     * (pred_conf-true_conf) * 5. +\
                      (1-obj_mask) * (pred_conf-true_conf) * ignore_mask
        cls_delta = obj_mask     * \
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=true_cls, logits=pred_cls)

        loss_xy = tf.reduce_sum(tf.square(xy_delta)) / num_batch
        loss_wh = tf.reduce_sum(tf.square(wh_delta)) / num_batch
        loss_conf = tf.reduce_sum(tf.square(conf_delta)) / num_batch
        loss_class = tf.reduce_sum(cls_delta) / num_batch

        loss = loss_xy + loss_wh + loss_conf + loss_class
        losses += loss

      else:
        stride = [img_w//grid_w,img_h//grid_h]
        pred_xcyc = pred_xcyc*stride
        pred_cls = tf.nn.sigmoid(logit[..., 5:])
        outputs.append(tf.concat([pred_xcyc, pred_wh, pred_conf, pred_cls], axis=-1))

    if is_training:
      return losses
    else:
      return outputs

  def pick_out_gt_box(self, y_true):
    y_true = y_true.copy()
    bs = y_true.shape[0]
    # print("=>y_true", y_true.shape)
    true_boxes_batch = np.zeros([bs, 1, 1, 1, self.MAX_OUTPUT_BOXES, 4], dtype=np.float32)
    # print("=>true_boxes_batch", true_boxes_batch.shape)
    for i in range(bs):
      y_true_per_layer = y_true[i]
      true_boxes_per_layer = y_true_per_layer[y_true_per_layer[..., 4] > 0][:, 0:4]
      if len(true_boxes_per_layer) == 0: continue
      true_boxes_batch[i][0][0][0][0:len(true_boxes_per_layer)] = true_boxes_per_layer

    return true_boxes_batch