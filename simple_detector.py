import numpy as np
import cv2
import glob
import os
from absl import app, flags

from tensorflow.keras.models import load_model
from core.image import draw_bboxes, preprocess_image, postprocess_bboxes, Shader, read_image


class Inference:
    def __init__(self, model_path):
        assert os.path.isdir(model_path), \
            "Model {} does not exist.".format(model_path)

        # Init tf model
        self.model = load_model(model_path)

    def predict(self, image, net_input_size):

        input_h, input_w = image.shape[:2]
        image = preprocess_image(
            image, (net_input_size, net_input_size)).astype(np.float32)
        output_h, output_w = image.shape[:2]

        images = np.expand_dims(image, axis=0)
        bboxes, scores, classes, valid_detections = self.model.predict(images)

        bboxes = bboxes[0][:valid_detections[0]]
        scores = scores[0][:valid_detections[0]]
        classes = classes[0][:valid_detections[0]]

        return postprocess_bboxes((output_w, output_h), (input_w, input_h), bboxes, scores, classes)


flags.DEFINE_string(
    'model', 'models/full_yolo4', 'Path to the keras full model (not only the weights)')
flags.DEFINE_integer('image_size', '416', 'CNN input image size')
flags.DEFINE_string('image_folder', '/data/Images/',
                    'Path to the image folder')
flags.DEFINE_multi_string('names', ['class1', 'class2'], 'class names')
FLAGS = flags.FLAGS


def main(_argv):
    model = Inference(FLAGS.model)

    shader = Shader(1)

    for img in glob.glob(FLAGS.image_folder+"*"):

        image = read_image(img)

        bboxes = model.predict(image, FLAGS.image_size)

        for bbox in bboxes:
            if bbox.score > 0.45:
                image = draw_bboxes(image, [bbox], FLAGS.names, shader)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image)

        if cv2.waitKey(0) == ord('q'):
            exit(0)


if __name__ == "__main__":
    app.run(main)
