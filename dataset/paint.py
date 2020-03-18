import cv2
import numpy as np


def transform_invert(bbox, img_size):
    """ Absolute to Relative
    """
    return np.concatenate([np.divide(bbox[..., 0:2], img_size), np.divide(bbox[..., 2:4], img_size)],
                          axis=-1).astype(np.float32)


def transform(bbox, img_size):
    """ Relative to Absolute
    """
    return np.concatenate([np.multiply(bbox[..., 0:2], img_size), np.multiply(bbox[..., 2:4], img_size)],
                          axis=-1).astype(np.int)


def draw_bbox(img, bbox, score, cls, color):
    msg = '{} {:.2f}%'.format(cls, score * 100)
    (x, y), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)

    bbox = tuple(bbox)
    img = cv2.rectangle(img, bbox[0:2], bbox[2:4], color, 5)
    img = cv2.rectangle(img, (bbox[0], bbox[1] - y - base),
                        (bbox[0] + x, bbox[1]),
                        color, -1)
    img = cv2.putText(img,
                      msg,
                      (bbox[0], bbox[1]),
                      cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
    return img


def draw_bboxes(img, bboxes, scores, clss, colors):
    for bbox, score, cls, color in zip(bboxes, scores, clss, colors):
        img = draw_bbox(img, bbox, score, cls, color)

    return img


def load_names(name_path):
    coco = {}
    with open(name_path, 'rt') as file:
        for index, label in enumerate(file):
            coco[index] = label.strip()

    return coco


def assign_name_and_color(idx, names):
    n = len(names)

    H = idx / n * 60

    def h2bgr(h):

        s = 0.9
        v = 0.9

        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c

        c = int(c * 255)
        x = int(x * 255)
        m = int(m * 255)

        i = int(h)
        if i == 0:
            return m, x + m, c + m
        elif i == 1:
            return m, c + m, x + m
        elif i == 2:
            return x + m, c + m, m
        elif i == 3:
            return c + m, x + m, m
        elif i == 4:
            return c + m, m, x + m
        elif i == 5:
            return x + m, m, c + m

    return names[idx], h2bgr(H)
