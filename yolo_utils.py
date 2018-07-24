# -*- coding: utf-8 -*-
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

def non_max_suppression(predictions, confidence_threshold=0.5, iou_threshold=0.5):
    """Applies Non-max suppression to prediction boxes.
       Most of codes come from https://github.com/mystic123/tensorflow-yolo-v3/blob/master/yolo_v3.py
    """
    predictions = np.asarray(predictions)
    conf_mask = np.expand_dims((predictions[:, :, 4] >= confidence_threshold), -1)
    predictions = predictions * conf_mask

    results = []
    for i, image_pred in enumerate(predictions):
      result = {}
      shape = image_pred.shape
      non_zero_idxs = np.nonzero(image_pred)
      image_pred = image_pred[non_zero_idxs]
      image_pred = image_pred.reshape(-1, shape[-1])

      bbox_attrs = image_pred[:, :5]
      classes = image_pred[:, 5:]
      classes = np.argmax(classes, axis=-1)

      unique_classes = list(set(classes.reshape(-1)))

      for cls in unique_classes:
        cls_mask = classes == cls
        cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
        cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
        cls_scores = cls_boxes[:, -1]
        cls_boxes = cls_boxes[:, :-1]

        while len(cls_boxes) > 0:
          box = cls_boxes[0]
          score = cls_scores[0]
          if not cls in result:
            result[cls] = []
          result[cls].append((box, score))
          cls_boxes = cls_boxes[1:]
          # update, remove bbox iou < iou_threshold
          ious = np.array([iou(box, x) for x in cls_boxes])
          iou_mask = ious < iou_threshold
          cls_boxes = cls_boxes[np.nonzero(iou_mask)]
          cls_scores = cls_scores[np.nonzero(iou_mask)]
      results.append(result)
    return results

def iou(box1, box2):
    """Computes Intersection over Union value for 2 bounding boxes
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    eps = 1e-5
    iou = int_area / (b1_area + b2_area - int_area + eps)
    return iou

def detections_boxes(detections):
  """Converts center x, center y, width and height values to coordinates of top
  left and bottom right points.
  """
  center_x, center_y, width, height, attrs = tf.split(detections,
                                                      [1, 1, 1, 1, -1], axis=-1)
  w2 = width / 2
  h2 = height / 2
  x0 = center_x - w2
  y0 = center_y - h2
  x1 = center_x + w2
  y1 = center_y + h2

  boxes = tf.concat([x0, y0, x1, y1], axis=-1)
  detections = tf.concat([boxes, attrs], axis=-1)
  return detections

def load_coco_name(path):
  """Load labels from coco.name
  """
  coco = {}
  with open(path, 'rt') as file:
    for index, label in enumerate(file):
      coco[index] = label.strip()

  return coco

def restore_saver():
  restore_dict = {}
  var_list = slim.get_model_variables()
  for var in var_list[:5]:
    restore_dict['Conv_0/'+var.name.split('/', 2)[-1].split(':')[0]] = var
  for var in var_list[5:]:
    restore_dict[var.name.split('/', 1)[-1].split(':')[0]] = var
  saver = tf.train.Saver(restore_dict)

  return saver