import os
import cv2
import tensorflow as tf
import numpy as np

from absl import app, flags
from tensorflow.keras import models

from core import config
from core.yolov3_tiny import YoloV3_Tiny


flags.DEFINE_string('path', None, 'Path to test list')

FLAGS = flags.FLAGS


def main(_argv):
    with open(FLAGS.path, 'r') as f:
        paths = [line.split(' ')[0] for line in f.readlines()]

    cfg = config.load('cfg/yymnist_yolov3-tiny.yaml')

    model = YoloV3_Tiny(cfg)
    model.summary()
    model.load_weights(cfg['test']['init_weight_path'])

    with open('./eval.txt', 'w') as f:
        for path in paths:
            img = cv2.imread(path)

            h, w = img.shape[:2]
            img = img[..., ::-1] / 255.

            imgs = np.expand_dims(img, axis=0)
            boxes, scores, classes, valid_detections = model.predict(imgs)

            for img, box, score, cls, valid in zip(imgs, boxes, scores, classes, valid_detections):
                valid_boxes = box[:valid]
                valid_score = score[:valid]
                valid_cls = cls[:valid]

                valid_boxes *= [w, h, w, h]

                line = ' '.join(['{:.4f},{:.4f},{:.4f},{:.4f},{:d},{:.4f}'.format(*_boxes, int(_cls), _score) for _boxes, _score, _cls in zip(valid_boxes, valid_score, valid_cls)])

            f.write(' '.join([path, line])+'\n')


if __name__ == '__main__':
    app.run(main)
