# -*- coding: utf-8 -*-
import cv2
import os

from absl import app, flags, logging

flags.DEFINE_string("src", None, "path to frame dir")
flags.DEFINE_string("dst", None, "path to output video")
flags.DEFINE_integer("fps", 25, "fps")
flags.DEFINE_string("format", "XVID", "codec used in VideoWriter when saving video to file")

FLAGS = flags.FLAGS


def main(_argv):
    path = FLAGS.src
    if os.path.isdir(path):
        paths = [os.path.join(path, name) for name in os.listdir(path)]

    first = cv2.imread(paths[0])
    h, w, _ = first.shape

    fps = FLAGS.fps
    codec = cv2.VideoWriter_fourcc(*FLAGS.format)
    logging.info("dst: {}, info: {}, {}, {}".format(FLAGS.dst, FLAGS.format, fps, (w, h)))
    out = cv2.VideoWriter(FLAGS.dst, codec, fps, (w, h))

    for path in paths:

        img = cv2.imread(path)
        img = cv2.resize(img, (w, h))

        cv2.imshow("output", img)
        if cv2.waitKey(1) == ord("q"):
            break

        out.write(img)


if __name__ == "__main__":
    app.run(main)
