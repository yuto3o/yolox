import numpy as np
import cv2
import glob
import os

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

model = Inference("./models/mAP-0.4702")

shader = Shader(1)
names = ["hole"]
image_size = 416

for img in glob.glob("/root/deep-learning-ws/data/Images/*"):

    image = read_image(img)

    bboxes = model.predict(image, image_size)

    for bbox in bboxes:
        if bbox.score > 0.45:
            image = draw_bboxes(image, [bbox], names, shader)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Image', image)

    if cv2.waitKey(0) == ord('q'):
        exit(0)
