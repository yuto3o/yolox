import yaml
import logging

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, Concatenate


from .darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, DarknetBlock
from .yolov3_commom import Yolo_Boxes, Yolo_NMS
from .utils import decode_yaml_tuple

with open('yolo/cfg/yolov3.yaml', 'r') as f:
    config = yaml.safe_load(f.read())

IOU_THRESHOLD = config['inference']['iou_threshold']
SCORE_THRESHOLD = config['inference']['score_threshold']
MAX_BOXES = config['inference']['max_boxes']

NUM_CLASSES = config['basic']['num_classes']
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

    tf.keras.backend.set_learning_phase(is_training)

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


