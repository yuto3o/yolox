# -*- coding: utf-8 -*-

from absl import app, flags
from core.utils import decode_cfg, load_weights
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, read_video, Shader

import time
import cv2
import numpy as np

flags.DEFINE_string('config', '', "path to config file")
flags.DEFINE_string('media', '',
                    'path to video file (MP4, AVI) or number for web camera(RTSP, or device ID) or image(JPEG, PNG)')
flags.DEFINE_bool('gpu', False, 'Use GPU')
FLAGS = flags.FLAGS


def main(_argv):
    # read config
    print('Config File From:', FLAGS.config)
    print('Media From:', FLAGS.media)
    print('Use GPU:', FLAGS.gpu)

    if not FLAGS.gpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    cfg = decode_cfg(FLAGS.config)

    model_type = cfg['yolo']['type']
    if model_type == 'yolov3':
        from core.model.one_stage.yolov3 import YOLOv3 as Model
    elif model_type == 'yolov3_tiny':
        from core.model.one_stage.yolov3 import YOLOv3_Tiny as Model
    elif model_type == 'yolov4':
        from core.model.one_stage.yolov4 import YOLOv4 as Model
    elif model_type == 'yolov4_tiny':
        from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model
    elif model_type == 'yolox':
        from core.model.one_stage.custom import YOLOX as Model
    else:
        raise NotImplementedError()

    _, model = Model(cfg)
    model.summary()

    init_weight_path = cfg['test']['init_weight_path']
    if init_weight_path:
        print('Load Weights File From:', init_weight_path)
        load_weights(model, init_weight_path)
    else:
        raise SystemExit('init_weight_path is Empty !')

    # assign colors for difference labels
    shader = Shader(cfg['yolo']['num_classes'])
    names = cfg['yolo']['names']
    image_size = cfg['test']['image_size'][0]

    def inference(image):

        h, w = image.shape[:2]
        image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
        images = np.expand_dims(image, axis=0)

        tic = time.time()
        bboxes, scores, classes, valid_detections = model.predict(images)
        toc = time.time()

        bboxes = bboxes[0][:valid_detections[0]]
        scores = scores[0][:valid_detections[0]]
        classes = classes[0][:valid_detections[0]]

        # bboxes *= image_size
        _, bboxes = postprocess_image(image, (w, h), bboxes)

        return (toc - tic) * 1000, bboxes, scores, classes

    if FLAGS.media.startswith('rtsp') or FLAGS.media.isdigit() or FLAGS.media.endswith(
            '.mp4') or FLAGS.media.endswith('.avi'):
        from collections import deque

        d = deque(maxlen=10)
        media = read_video(FLAGS.media)

        while True:

            ret, image = media.read()
            if not ret: break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ms, bboxes, scores, classes = inference(image)
            image = draw_bboxes(image, bboxes, scores, classes, names, shader)
            d.append(ms)

            mms = np.mean(d)
            print('Inference Time:', mms, 'ms')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.putText(image,
                                "{:.2f} ms".format(mms),
                                (0, 30),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            cv2.imshow('Image', image)
            if cv2.waitKey(33) == ord('q'):
                break

        media.release()

    elif FLAGS.media.endswith('.jpg') or FLAGS.media.endswith('.png'):
        image = read_image(FLAGS.media)

        ms, bboxes, scores, classes = inference(image)
        image = draw_bboxes(image, bboxes, scores, classes, names, shader)

        print('Inference Time:', ms, 'ms')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image)
        cv2.waitKey()


if __name__ == '__main__':
    app.run(main)
