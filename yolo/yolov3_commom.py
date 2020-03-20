# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add, Layer
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.regularizers import l2

import tensorflow as tf

WEIGHT_DECAY = 5e-4
LEAKY_ALPHA = 0.1


def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {"kernel_regularizer": l2(WEIGHT_DECAY), "padding": "valid" if kwargs.get(
        "strides") == (2, 2) else "same"}
    darknet_conv_kwargs.update(kwargs)

    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    without_bias_kwargs = {"use_bias": False}
    without_bias_kwargs.update(kwargs)

    def wrapper(x):
        x = DarknetConv2D(*args, **without_bias_kwargs)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=LEAKY_ALPHA)(x)
        return x

    return wrapper


def DarknetBlock(filters, niter):
    def wrapper(x):
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        x = DarknetConv2D_BN_Leaky(filters, (3, 3), strides=(2, 2))(x)
        for _ in range(niter):
            y = DarknetConv2D_BN_Leaky(filters // 2, (1, 1))(x)
            y = DarknetConv2D_BN_Leaky(filters, (3, 3))(y)
            x = Add()([x, y])
        return x

    return wrapper


class UpSampleLike(Layer):

    def __init__(self, y, **kwargs):
        self.target_shape = y.shape.as_list()[1:3]
        super(UpSampleLike, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(UpSampleLike, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.image.resize(x, self.target_shape, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.target_shape[0], self.target_shape[1], input_shape[-1]


def Yolo_Boxes(num_classes, anchors, stride, name=None):
    return Lambda(lambda x: _yolo_boxes(x, num_classes, anchors, stride), name=name)


def _yolo_boxes(logits, num_classes, anchors, stride):
    """logits: (batch_size, grid, grid, num_anchors*(5 + num_classes))
    """
    x_shape = tf.shape(logits)

    logits = tf.reshape(logits, (x_shape[0], x_shape[1], x_shape[2], anchors.shape[0], num_classes + 5))
    anchors = tf.cast(anchors, tf.float32)

    grid_shape = x_shape[1:3]
    grid_h, grid_w = grid_shape[0], grid_shape[1]

    anchors /= tf.cast(tf.multiply([grid_w, grid_h], stride), tf.float32)

    box_xy, box_wh, obj, cls = tf.split(
        logits, (2, 2, 1, num_classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    obj = tf.sigmoid(obj)
    cls = tf.sigmoid(cls)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast([grid_w, grid_h], tf.float32)
    box_wh = tf.exp(box_wh) * tf.cast(anchors, tf.float32)

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    box = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return box, obj, cls, pred_box


def Yolo_NMS(max_boxes, iou_threshold, score_threshold, name=None):
    return Lambda(lambda x: _yolo_nms(x, max_boxes, iou_threshold, score_threshold), name=name)


def _yolo_nms(inputs, max_boxes, iou_threshold, score_threshold):
    box, obj, cls = [], [], []

    for _box, _obj, _cls in inputs:
        batch = tf.shape(_box)[0]
        _box = tf.reshape(_box, [batch, -1, 4])
        _obj = tf.reshape(_obj, [batch, -1, 1])
        num_classes = tf.shape(_cls)[-1]
        _cls = tf.reshape(_cls, [batch, -1, num_classes])

        box.append(_box)
        obj.append(_obj)
        cls.append(_cls)

    box = tf.concat(box, axis=1)
    obj = tf.concat(obj, axis=1)
    cls = tf.concat(cls, axis=1)

    scores = obj * cls
    boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(box, (tf.shape(box)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_boxes,
        max_total_size=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )

    return boxes, scores, classes, valid


def Yolo_Loss(num_classes, anchors, stride, ignore_thresh):
    def wrapper(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, num_anchors*(5 + num_classes))
        pred_box, pred_obj, pred_class, pred_xywh = _yolo_boxes(
            y_pred, num_classes, anchors, stride)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class = tf.split(y_true, (4, 1, num_classes), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_h, grid_w = tf.shape(y_true)[1:3]
        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast([grid_w, grid_h], tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(_broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * binary_crossentropy(true_class, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss

    return wrapper


def _broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)
