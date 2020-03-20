# -*- coding: utf-8 -*-
import cv2
import os
import time
import tensorflow  as tf

from absl import app, flags, logging
from yolo.yolov3 import YoloV3
from yolo.yolov3_tiny import YoloV3_Tiny
from yolo.utils import preprocess_image
from yolo import config
from tools.paint import draw_bboxes

flags.DEFINE_string("config", None, "path to config file")
flags.DEFINE_boolean("tiny", False, "yolov3 or yolov3-tiny")
flags.DEFINE_list("size", [960, 512], "resize images(keep aspect ratio)")
flags.DEFINE_string("src", None, "path to video file or number for web camera")
flags.DEFINE_string("dst", None, "path to output video")
flags.DEFINE_string("format", "XVID", "codec used in VideoWriter when saving video to file")

FLAGS = flags.FLAGS


def main(_argv):
    path = FLAGS.config
    if path and os.path.exists(path):
        cfg = config.load(path)
    else:
        raise FileNotFoundError("config file is not existed: %s" % path)
    names_list = cfg["yolo"]["classes"]

    if FLAGS.tiny:
        yolo = YoloV3_Tiny(cfg)
    else:
        yolo = YoloV3(cfg)

    yolo.load_weights(cfg["test"]["init_weights"])
    logging.info("weights loaded")

    try:
        vid = cv2.VideoCapture(int(FLAGS.src))
    except:
        vid = cv2.VideoCapture(FLAGS.src)

    out = None

    if FLAGS.dst:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.format)
        logging.info("dst: {}, info: {}, {}, {}".format(FLAGS.dst, FLAGS.format, fps, FLAGS.size))
        out = cv2.VideoWriter(FLAGS.dst, codec, fps, tuple(FLAGS.size))

    times = []
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img = preprocess_image(FLAGS.size, img)
        img_in = tf.expand_dims(img, 0)

        t1 = time.time()
        boxes, scores, classes, valid_detections = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2 - t1)
        times = times[-20:]
        fps = len(times) / sum(times)
        ms = 1 / fps * 1000
        msg = "{:.2f} fps, {:.2f} ms".format(fps, ms)
        logging.info(msg)

        box, score, cls, valid = boxes[0], scores[0], classes[0], valid_detections[0]
        valid_boxes = box[:valid]
        valid_score = score[:valid]
        valid_cls = cls[:valid]

        img = draw_bboxes(img, valid_boxes, valid_score, valid_cls, names_list)
        img = cv2.putText(img, msg, (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

        if FLAGS.dst:
            out.write(img)

        cv2.imshow("output", img)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
# python sensor.py --src=./mvi39311.mp4 --dst=./mvi3911_sensor.mp4 --config=./cfg/yolov3.yaml
