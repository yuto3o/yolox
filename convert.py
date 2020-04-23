# -*- coding: utf-8 -*-
from absl import app, flags
from core.parser import YoloParser
from core import config

flags.DEFINE_bool("tiny", False, "if core-tiny")
flags.DEFINE_string("config", "./checkpoints/yolov3/yolov3.cfg", "path to yolov3.cfg")
flags.DEFINE_string("weights", "./checkpoints/yolov3/yolov3.weights", "path to yolov3.weights")
flags.DEFINE_string("output", "./yolov3", "path to output file")

FLAGS = flags.FLAGS


def main(_argv):
    if FLAGS.tiny:
        from core.yolov3_tiny import YoloV3_Tiny

        cfg = config.load("./cfg/yolov3-tiny.yaml")

        YoloParser(YoloV3_Tiny(cfg),
                   FLAGS.config,
                   FLAGS.weights,
                   FLAGS.output).run()
    else:
        from core.yolov3 import YoloV3

        cfg = config.load("./cfg/yolov3.yaml")

        YoloParser(YoloV3(cfg),
                   FLAGS.config,
                   FLAGS.weights,
                   FLAGS.output).run()


if __name__ == "__main__":
    app.run(main)
