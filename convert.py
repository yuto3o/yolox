# -*- coding: utf-8 -*-
from absl import app, flags
from yolo.adapter import YoloParser
from yolo import config

flags.DEFINE_bool("tiny", False, "if yolo-tiny")
flags.DEFINE_string("config", "./checkpoints/yolov3/yolov3.cfg", "path to yolov3.cfg")
flags.DEFINE_string("weights", "./checkpoints/yolov3/yolov3.weights", "path to yolov3.weights")
flags.DEFINE_string("output", "./yolov3.h5", "path to output file")

FLAGS = flags.FLAGS


def main(_argv):
    if FLAGS.tiny:
        from yolo.yolov3_tiny import YoloV3_Tiny

        cfg = config.load("cfg/yolov3-tiny.yaml")

        YoloParser(YoloV3_Tiny(cfg),
                   FLAGS.config,
                   FLAGS.weights,
                   FLAGS.output).run()
    else:
        from yolo.yolov3 import YoloV3

        cfg = config.load("cfg/yolov3.yaml")

        YoloParser(YoloV3(cfg),
                   FLAGS.config,
                   FLAGS.weights,
                   FLAGS.output).run()


if __name__ == "__main__":
    app.run(main)
