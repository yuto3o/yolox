import cv2
import numpy as np


def draw_bboxes(img, bboxes, scores, clss, names_list):
    """ Draw boxes on the img
    :param img: H x W x 3, RGB, value in [0, 1], float32
    :param bboxes: N x 4, [x1, y1, x2, y2]
    :param scores: N,
    :param clss: N,
    :param names_list: N, str
    :return: H x W x 3
    """

    def draw_bbox(img, bbox, score, cls):
        name, color = assign_name_and_color(cls, names_list)

        msg = '{} {:.2f}%'.format(name, score * 100)
        (x, y), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)

        h, w, _ = img.shape
        bbox = transform(bbox, (w, h))
        bbox = tuple(bbox)

        img = cv2.rectangle(img, bbox[0:2], bbox[2:4], color, 2)
        img = cv2.rectangle(img, (bbox[0], bbox[1] - y - base),
                            (bbox[0] + x, bbox[1]),
                            color, -1)
        img = cv2.putText(img,
                          msg,
                          (bbox[0], bbox[1]),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)
        return img

    img = img[:, :, ::-1]
    img *= 255
    img = np.array(img, np.uint8)

    for bbox, score, cls in zip(bboxes, scores, clss):
        img = draw_bbox(img, bbox, score, cls)

    return img


def transform(bbox, img_size):
    """ Relative to Absolute
    """
    w, h = img_size
    return np.array([bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]).astype(np.int)


def assign_name_and_color(idx, names):
    n = len(names)
    idx = int(idx)

    H = idx / n * 6

    def h2rgb(h):

        s = 0.7
        v = 0.7

        c = v * s
        x = c * (1 - abs(h % 2 - 1))
        m = v - c

        c = int(c * 255)
        x = int(x * 255)
        m = int(m * 255)

        i = int(h)
        if i == 0:
            return c + m, x + m, m
        elif i == 1:
            return x + m, c + m, m
        elif i == 2:
            return m, c + m, x + m
        elif i == 3:
            return m, x + m, c + m
        elif i == 4:
            return x + m, m, c + m
        elif i == 5:
            return c + m, m, x + m

    return names[idx], h2rgb(H)
