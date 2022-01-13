# -*- coding: utf-8 -*-
import numpy as np
import cv2


class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax, score, cls):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.score = score
        self.cls = int(cls)


def read_image(*args, **kwargs):
    return cv2.cvtColor(cv2.imread(*args, **kwargs), cv2.COLOR_BGR2RGB)


def read_video(*args, **kwargs):
    return cv2.VideoCapture(*args, **kwargs)


def postprocess_bboxes(input_size, output_size, bboxes, scores, classes):
    input_w, input_h = input_size
    output_w, output_h = output_size

    scale = min(input_w / output_w, input_h / output_h)
    nw, nh = int(scale * output_w), int(scale * output_h)
    dw, dh = (input_w - nw) // 2, (input_h - nh) // 2

    bboxes = bboxes.astype(np.float32)
    bboxes[:, [0, 2]] = np.clip((bboxes[:, [0, 2]] - dw) / scale, 0., output_w)
    bboxes[:, [1, 3]] = np.clip((bboxes[:, [1, 3]] - dh) / scale, 0., output_h)

    boxes = []
    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i][:4]
        boxes.append(BoundingBox(x1, y1, x2, y2, scores[i], classes[i]))

    return boxes


def postprocess_image(image, size, bboxes=None):
    """
    :param image: RGB, uint8
    :param size:
    :param bboxes:
    :return: RGB, uint8
    """
    ih, iw = image.shape[:2]
    w, h = size

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image = image[dh:nh + dh, dw:nw + dw, :]
    image_resized = cv2.resize(image, (w, h))

    if bboxes is None:
        return image_resized
    else:
        bboxes = bboxes.astype(np.float32)
        bboxes[:, [0, 2]] = np.clip((bboxes[:, [0, 2]] - dw) / scale, 0., w)
        bboxes[:, [1, 3]] = np.clip((bboxes[:, [1, 3]] - dh) / scale, 0., h)

        return image_resized, bboxes


def preprocess_image(image, size, bboxes=None):
    """
    :param image: RGB, uint8
    :param size:
    :param bboxes:
    :return: RGB, uint8
    """
    iw, ih = size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], dtype=np.uint8, fill_value=127)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized

    if bboxes is None:
        return image_paded

    else:
        bboxes = np.asarray(bboxes).astype(np.float32)
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh

        return image_paded, bboxes
