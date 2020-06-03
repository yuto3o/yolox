# -*- coding: utf-8 -*-
import tensorflow as tf
import math
EPS = 1e-8

def GIoU(y_pred_box, y_true_box):
    """
    Calculate GIoU loss on anchor boxes
    Reference Paper:
        "Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression"
        https://arxiv.org/abs/1902.09630
    Parameters
    ----------
    y_pred_box: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), x1y1x2y2
    y_true_box: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), x1y1x2y2
    Returns
    -------
    giou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    y_pred_box_min = y_pred_box[..., :2]
    y_pred_box_max = y_pred_box[..., 2:4]
    y_pred_box_wh = y_pred_box_max - y_pred_box_min

    y_true_box_min = y_true_box[..., :2]
    y_true_box_max = y_true_box[..., 2:4]
    y_true_box_wh = y_true_box_max - y_true_box_min

    intersect_min = tf.maximum(y_true_box_min, y_pred_box_min)
    intersect_max = tf.minimum(y_true_box_max, y_pred_box_max)
    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    y_pred_box_area = y_pred_box_wh[..., 0] * y_pred_box_wh[..., 1]
    y_true_box_area = y_true_box_wh[..., 0] * y_true_box_wh[..., 1]
    union_area = y_pred_box_area + y_true_box_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / tf.maximum(union_area, EPS)

    # get enclosed area
    enclose_min = tf.minimum(y_true_box_min, y_pred_box_min)
    enclose_max = tf.maximum(y_true_box_max, y_pred_box_max)
    enclose_wh = tf.maximum(enclose_max - enclose_min, 0.0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    # calculate GIoU, add epsilon in denominator to avoid dividing by 0
    giou = iou - (enclose_area - union_area) / tf.maximum(enclose_area, EPS)

    return giou


def DIoU(y_pred_box, y_true_box):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287
    Parameters
    ----------
    y_pred_box: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), x1y1x2y2
    y_true_box: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), x1y1x2y2
    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    y_pred_box_min = y_pred_box[..., :2]
    y_pred_box_max = y_pred_box[..., 2:4]
    y_pred_box_wh = y_pred_box_max - y_pred_box_min
    y_pred_box_center = (y_pred_box_min + y_pred_box_max) / 2.

    y_true_box_min = y_true_box[..., :2]
    y_true_box_max = y_true_box[..., 2:4]
    y_true_box_wh = y_true_box_max - y_true_box_min
    y_true_box_center = (y_true_box_min + y_true_box_max) / 2.

    intersect_min = tf.maximum(y_pred_box_min, y_true_box_min)
    intersect_max = tf.minimum(y_pred_box_max, y_true_box_max)
    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    y_true_box_area = y_true_box_wh[..., 0] * y_true_box_wh[..., 1]
    y_pred_box_area = y_pred_box_wh[..., 0] * y_pred_box_wh[..., 1]
    union_area = y_true_box_area + y_pred_box_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / tf.maximum(union_area, EPS)

    # box center distance
    center_distance = tf.reduce_sum(tf.square(y_pred_box_center - y_true_box_center), axis=-1)
    # get enclosed area
    enclose_min = tf.minimum(y_pred_box_min, y_true_box_min)
    enclose_max = tf.maximum(y_pred_box_max, y_true_box_max)
    enclose_wh = tf.maximum(enclose_max - enclose_min, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - center_distance / tf.maximum(enclose_diagonal, EPS)

    return diou


def CIoU(y_pred_box, y_true_box):
    """
    Calculate DIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287
    Parameters
    ----------
    y_pred_box: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), x1y1x2y2
    y_true_box: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), x1y1x2y2
    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    y_pred_box_min = y_pred_box[..., :2]
    y_pred_box_max = y_pred_box[..., 2:4]
    y_pred_box_wh = y_pred_box_max - y_pred_box_min
    y_pred_box_center = (y_pred_box_min + y_pred_box_max) / 2.

    y_true_box_min = y_true_box[..., :2]
    y_true_box_max = y_true_box[..., 2:4]
    y_true_box_wh = y_true_box_max - y_true_box_min
    y_true_box_center = (y_true_box_min + y_true_box_max) / 2.

    intersect_min = tf.maximum(y_pred_box_min, y_true_box_min)
    intersect_max = tf.minimum(y_pred_box_max, y_true_box_max)
    intersect_wh = tf.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    y_true_box_area = y_true_box_wh[..., 0] * y_true_box_wh[..., 1]
    y_pred_box_area = y_pred_box_wh[..., 0] * y_pred_box_wh[..., 1]
    union_area = y_true_box_area + y_pred_box_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / tf.maximum(union_area, EPS)

    # box center distance
    center_distance = tf.reduce_sum(tf.square(y_pred_box_center - y_true_box_center), axis=-1)
    # get enclosed area
    enclose_min = tf.minimum(y_pred_box_min, y_true_box_min)
    enclose_max = tf.maximum(y_pred_box_max, y_true_box_max)
    enclose_wh = tf.maximum(enclose_max - enclose_min, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - center_distance / tf.maximum(enclose_diagonal, EPS)

    # calculate param v and alpha to extend to CIoU
    constant = 4. / (math.pi * math.pi)
    v = constant * tf.square(
        tf.math.atan2(y_true_box_wh[..., 0], tf.maximum(y_true_box_wh[..., 1], EPS)) - tf.math.atan2(
            y_pred_box_wh[..., 0], tf.maximum(y_pred_box_wh[..., 1], EPS)))
    alpha = v / tf.maximum(1.0 - iou + v, EPS)
    ciou = diou - alpha * v
    return ciou
