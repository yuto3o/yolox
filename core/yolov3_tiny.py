# -*- coding: utf-8 -*-
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, Concatenate, MaxPool2D
from .yolov3_commom import Yolo_Boxes, Yolo_NMS, DarknetConv2D, DarknetConv2D_BN_Leaky


def YoloV3_Tiny(cfg, is_training=False, name="yolov3_tiny"):
    IOU_THRESHOLD = cfg["yolo"]["iou_threshold"]
    SCORE_THRESHOLD = cfg["yolo"]["score_threshold"]
    MAX_BOXES = cfg["yolo"]["max_boxes"]
    NUM_CLASSES = cfg["yolo"]["num_classes"]
    STRIDES = cfg["yolo"]["strides"]
    MASK = cfg["yolo"]["mask"]
    ANCHOR = cfg["yolo"]["anchors"]

    tf.keras.backend.set_learning_phase(is_training)

    x = inputs = Input([None, None, 3])

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

    output_0 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[0]], STRIDES[0], "yolo_boxes_0")(output_0)
    output_1 = Yolo_Boxes(NUM_CLASSES, ANCHOR[MASK[1]], STRIDES[1], "yolo_boxes_1")(output_1)

    # boxes, scores, classes, valid_detections
    outputs = Yolo_NMS(MAX_BOXES, IOU_THRESHOLD, SCORE_THRESHOLD, "yolo_nms")([output_0[:3], output_1[:3]])

    return Model(inputs, outputs, name=name)
