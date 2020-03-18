import yaml
import logging

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, Concatenate, Lambda

from .darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetBlock
from .utils import decode_yaml_tuple

with open('yolo/config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())

IOU_THRESHOLD = config['inference']['iou_threshold']
SCORE_THRESHOLD = config['inference']['score_threshold']
NUM_CLASSES = config['inference']['num_classes']
MAX_BOXES = config['inference']['max_boxes']

IMAGE_SIZE = config['basic']['image_size']
ANCHOR = config['basic']['anchors']
MASK = config['basic']['mask']

msg = "Loading YoloV3 Config: {} = {}"
logging.info(msg.format('iou_threshold', IOU_THRESHOLD))
logging.info(msg.format('score_threshold', SCORE_THRESHOLD))
logging.info(msg.format('num_classes', NUM_CLASSES))
logging.info(msg.format('max_boxes', MAX_BOXES))

logging.info(msg.format('image_size', IMAGE_SIZE))
logging.info(msg.format('anchors', ANCHOR))
logging.info(msg.format('mask', MASK))

ANCHOR = decode_yaml_tuple(ANCHOR).astype('float32')
ANCHOR /= IMAGE_SIZE
MASK = decode_yaml_tuple(MASK)


def YoloV3(is_training=False, name="yolov3"):
    x = inputs = Input([416, 416, 3])

    x = DarknetConv2D_BN_Leaky(32, 3)(x)  # 0
    x = DarknetBlock(64, 1)(x)  # 1-4
    x = DarknetBlock(128, 2)(x)  # 5-11
    x = x_36 = DarknetBlock(256, 8)(x)  # 12-36
    x = x_61 = DarknetBlock(512, 8)(x)  # 37-61
    x = DarknetBlock(1024, 4)(x)  # 62-74

    x = DarknetConv2D_BN_Leaky(512, 1)(x)  # 75
    x = DarknetConv2D_BN_Leaky(1024, 3)(x)  # 76
    x = DarknetConv2D_BN_Leaky(512, 1)(x)  # 77
    x = DarknetConv2D_BN_Leaky(1024, 3)(x)  # 78
    x = x_79 = DarknetConv2D_BN_Leaky(512, 1)(x)  # 79

    x = DarknetConv2D_BN_Leaky(1024, 3)(x)  # 80
    output_0 = DarknetConv2D(len(MASK[0]) * (NUM_CLASSES + 5), 1)(x)

    x = DarknetConv2D_BN_Leaky(256, 1)(x_79)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_61])

    x = DarknetConv2D_BN_Leaky(256, 1)(x)
    x = DarknetConv2D_BN_Leaky(512, 3)(x)
    x = DarknetConv2D_BN_Leaky(256, 1)(x)
    x = DarknetConv2D_BN_Leaky(512, 3)(x)
    x = x_91 = DarknetConv2D_BN_Leaky(256, 1)(x)

    x = DarknetConv2D_BN_Leaky(512, 3)(x)
    output_1 = DarknetConv2D(len(MASK[1]) * (NUM_CLASSES + 5), 1)(x)

    x = DarknetConv2D_BN_Leaky(128, 1)(x_91)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_36])

    x = DarknetConv2D_BN_Leaky(128, 1)(x)
    x = DarknetConv2D_BN_Leaky(256, 3)(x)
    x = DarknetConv2D_BN_Leaky(128, 1)(x)
    x = DarknetConv2D_BN_Leaky(256, 3)(x)
    x = DarknetConv2D_BN_Leaky(128, 1)(x)
    x = DarknetConv2D_BN_Leaky(256, 3)(x)

    output_2 = DarknetConv2D(len(MASK[2]) * (NUM_CLASSES + 5), 1)(x)

    if is_training:
        return Model(inputs, (output_0, output_1, output_2), name=name)

    output_2 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[2]], "yolo_boxes_2")(output_2)
    output_1 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[1]], "yolo_boxes_1")(output_1)
    output_0 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[0]], "yolo_boxes_0")(output_0)

    # boxes, scores, classes, valid_detections
    outputs = Yolo_NMS(MAX_BOXES, IOU_THRESHOLD, SCORE_THRESHOLD, "yolo_nms")(
        [output_0[:3], output_1[:3], output_2[:3]])

    return Model(inputs, outputs, name=name)


def Yolo_Boxes(num_classes, anchors, name=None):
    def wrapper(logits):
        """logits: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        """
        x_shape = tf.shape(logits)
        logits = Lambda(
            lambda x: tf.reshape(x, (x_shape[0], x_shape[1], x_shape[2], len(anchors), num_classes + 5)))(logits)

        grid_shape = x_shape[1:3]
        grid_h, grid_w = grid_shape[0], grid_shape[1]
        box_xy, box_wh, confi, probs = tf.split(
            logits, (2, 2, 1, num_classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        confi = tf.sigmoid(confi)
        probs = tf.sigmoid(probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast([grid_w, grid_h], tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        batch = x_shape[0]
        bbox = tf.reshape(bbox, [batch, -1, 4])
        confi = tf.reshape(confi, [batch, -1, 1])
        probs = tf.reshape(probs, [batch, -1, num_classes])
        pred_box = tf.reshape(pred_box, [batch, -1, 4])

        return bbox, confi, probs, pred_box

    return Lambda(wrapper, name=name)


def Yolo_NMS(max_boxes, iou_threshold, score_threshold, name=None):
    def wrapper(inputs):
        # boxes, conf, type
        bbox, confi, probs = [], [], []

        for _bbox, _confi, _probs in inputs:
            bbox.append(_bbox)
            confi.append(_confi)
            probs.append(_probs)

        bbox = tf.concat(bbox, axis=1)
        confi = tf.concat(confi, axis=1)
        probs = tf.concat(probs, axis=1)

        scores = confi * probs
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(
                scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=max_boxes,
            max_total_size=max_boxes,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        return boxes, scores, classes, valid_detections

    return Lambda(wrapper, name=name)
