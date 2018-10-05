# -*- coding: utf-8 -*-
import tensorflow as tf
slim = tf.contrib.slim

import numpy as np

def preprocess_true_boxes(true_boxes, input_size, anchors, num_classes):
  '''Preprocess true boxes to training input format
  Params:
    true_boxes: array, shape=(m, T, 5)
        Absolute xc, yc, w, h, class_id relative to input_shape.
    input_size: array-like, (w, h), multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
  Returns:
    labels: list of array, shape like yolo_outputs, xywh are reletive value

  Codes come from https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py
  '''
  assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
  num_layers = len(anchors)//3 # default setting
  anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

  true_boxes = np.array(true_boxes, dtype='float32')
  input_size = np.array(input_size, dtype='int32')
  boxes_xy = true_boxes[..., 0:2]
  boxes_wh = true_boxes[..., 2:4]
  true_boxes[..., 0:2] = boxes_xy/input_size
  true_boxes[..., 2:4] = boxes_wh/input_size

  m = true_boxes.shape[0]
  grid_sizes = [input_size//stride for stride in [8, 16, 32]]
  labels = [np.zeros((m,grid_sizes[l][1],grid_sizes[l][0],len(anchor_mask[l]),5+num_classes),
      dtype='float32') for l in range(num_layers)]

  # Expand dim to apply broadcasting.
  anchors = np.expand_dims(anchors, 0)
  anchor_maxes = anchors / 2.
  anchor_mins = -anchor_maxes
  valid_mask = boxes_wh[..., 0]>0

  for b in range(m):
    # Discard zero rows.
    wh = boxes_wh[b, valid_mask[b]]
    if len(wh)==0: continue
    # Expand dim to apply broadcasting.
    wh = np.expand_dims(wh, -2)
    box_maxes = wh / 2.
    box_mins = -box_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    box_area = wh[..., 0] * wh[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box
    best_anchor = np.argmax(iou, axis=-1)

    for t, n in enumerate(best_anchor):
      for l in range(num_layers):
        if n in anchor_mask[l]:
          i = np.floor(true_boxes[b,t,0]*grid_sizes[l][0]).astype('int32')
          j = np.floor(true_boxes[b,t,1]*grid_sizes[l][1]).astype('int32')
          k = anchor_mask[l].index(n)
          c = true_boxes[b, t, 4].astype('int32')
          labels[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
          labels[l][b, j, i, k, 4] = 1
          labels[l][b, j, i, k, 5+c] = 1

  labels = tf.concat([tf.reshape(labels[l],
            [m,grid_sizes[l][1]*grid_sizes[l][0]*len(anchor_mask[l]),5+num_classes]) for l in range(num_layers)], axis=1)
  return labels

def non_max_suppression(predictions, confidence_threshold=0.5, iou_threshold=0.5):
    """Applies Non-max suppression to prediction boxes.
       Most of codes come from
         https://github.com/mystic123/tensorflow-yolo-v3/blob/master/yolo_v3.py
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
          ious = np.array([box_iou(box, x) for x in cls_boxes])
          print(ious)
          iou_mask = ious < iou_threshold
          cls_boxes = cls_boxes[np.nonzero(iou_mask)]
          cls_scores = cls_scores[np.nonzero(iou_mask)]
      results.append(result)
    return results

def yolo_boxes_to_corners(bbox):
  """Convert xc, yc, w, h to xmin, ymin, xmax, ymax along last axis
  """

  bbox_attr = bbox[..., 0:4]
  bbox_min = bbox_attr[...,0:2] - bbox_attr[...,2:4]/2.
  bbox_max = bbox_attr[...,0:2] + bbox_attr[...,2:4]/2.
  bbox_attr = np.concatenate([bbox_min, bbox_max], axis=-1)
  bbox = np.concatenate([bbox_attr, bbox[..., 4::]], axis=-1)
  return bbox


def box_iou(box1, box2):
  """Computes Intersection over Union value for 2 bounding boxes

  Params:
    xc, yc, w, h
  """
  box1_xyxy = yolo_boxes_to_corners(box1)
  box2_xyxy = yolo_boxes_to_corners(box2)

  intersect_min = np.maximum(box1_xyxy[...,0:2], box2_xyxy[...,0:2])
  intersect_max = np.minimum(box1_xyxy[...,2:4], box2_xyxy[...,2:4])
  intersect_wh = np.maximum(intersect_max-intersect_min, 0)
  intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

  bbox1_area = box1[..., 2]*box1[..., 3]
  bbox2_area = box2[..., 2]*box2[..., 3]

  eps = 1e-5
  iou = intersect_area/(bbox1_area+bbox2_area-intersect_area+eps)
  return iou

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