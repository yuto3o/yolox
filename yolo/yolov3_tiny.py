import yaml
import logging

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, Concatenate, MaxPool2D


from .darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetBlock
from .yolov3_commom import Yolo_Boxes, Yolo_NMS
from .utils import decode_yaml_tuple

with open('yolo/cfg/yolov3_tiny.yaml', 'r') as f:
    config = yaml.safe_load(f.read())

IOU_THRESHOLD = config['inference']['iou_threshold']
SCORE_THRESHOLD = config['inference']['score_threshold']
MAX_BOXES = config['inference']['max_boxes']

NUM_CLASSES = config['basic']['num_classes']
IMAGE_SIZE = config['basic']['image_size']
ANCHOR = config['basic']['anchors']
MASK = config['basic']['mask']

msg = "Loading YoloV3-Tiny Config: {} = {}"
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


def YoloV3_Tiny(is_training=False, name="yolov3_tiny"):

    tf.keras.backend.set_learning_phase(is_training)

    x = inputs = Input([416, 416, 3])

    x = DarknetConv2D_BN_Leaky(16, 3)(x)
    x = MaxPool2D(2, 2, "same")(x)
    x = DarknetConv2D_BN_Leaky(32, 3)(x)
    x = MaxPool2D(2, 2, "same")(x)
    x = DarknetConv2D_BN_Leaky(64, 3)(x)
    x = MaxPool2D(2, 2, "same")(x)
    x = DarknetConv2D_BN_Leaky(128, 3)(x)
    x = MaxPool2D(2, 2, "same")(x)
    x = x_8 = DarknetConv2D_BN_Leaky(256, 3)(x)
    x = MaxPool2D(2, 2, "same")(x)
    x = DarknetConv2D_BN_Leaky(512, 3)(x)
    x = MaxPool2D(2, 1, "same")(x)
    x = DarknetConv2D_BN_Leaky(1024, 3)(x)

    x = x_13 = DarknetConv2D_BN_Leaky(256, 1)(x)

    x = DarknetConv2D_BN_Leaky(512, 3)(x)
    output_0 = DarknetConv2D(len(MASK[0]) * (NUM_CLASSES + 5), 1)(x)

    x = DarknetConv2D_BN_Leaky(128, 1)(x_13)
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, x_8])

    x = DarknetConv2D_BN_Leaky(256, 3)(x)
    output_1 = DarknetConv2D(len(MASK[1]) * (NUM_CLASSES + 5), 1)(x)

    if is_training:
        return Model(inputs, (output_0, output_1), name=name)

    output_1 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[1]], "yolo_boxes_1")(output_1)
    output_0 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[0]], "yolo_boxes_0")(output_0)

    # boxes, scores, classes, valid_detections
    outputs = Yolo_NMS(MAX_BOXES, IOU_THRESHOLD, SCORE_THRESHOLD, "yolo_nms")(
        [output_0[:3], output_1[:3]])

    return Model(inputs, outputs, name=name)