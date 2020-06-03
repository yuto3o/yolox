# -*- coding: utf-8 -*-
import cv2


class Shader:

    def __init__(self, num_colors):

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

        self._colors = {i: h2rgb(i / num_colors * 6) for i in range(num_colors)}

    def get_color(self, index):
        """
        :param index: int
        :return: (R, G, B), uint8
        """
        return self._colors[index]


def draw_bboxes(img, bboxes, scores, classes, names, shader, type='absolute'):
    if type == 'absolute':
        return _draw_bboxes_absolute(img, bboxes, scores, classes, names, shader)
    elif type == 'relative':
        return _draw_bboxes_relative(img, bboxes, scores, classes, names, shader)
    else:
        raise NotImplementedError()


def _draw_bboxes_relative(img, bboxes, scores, classes, names, shader):
    """
   BBoxes is relative Format
   :param img: BGR, uint8
   :param bboxes: x1, y1, x2, y2, float
   :return: img, BGR, uint8
   """

    def _draw_bbox(img, bbox, score, cls):
        msg = '{} {:.2%}'.format(names[int(cls)], score)
        (x, y), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
        color = shader.get_color(int(cls))

        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.rectangle(img, (x1, y1 - y - base),
                            (x1 + x, y1),
                            color, -1)
        img = cv2.putText(img,
                          msg,
                          (x1, y1),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

        return img

    for bbox, score, cls in zip(bboxes, scores, classes):
        img = _draw_bbox(img, bbox, score, cls)

    return img


def _draw_bboxes_absolute(img, bboxes, scores, classes, names, shader):
    """
    BBoxes is absolute Format
    :param img: BGR, uint8
    :param bboxes: x1, y1, x2, y2, int
    :return: img, BGR, uint8
    """

    def _draw_bbox(img, bbox, score, cls):
        msg = '{} {:.2%}'.format(names[int(cls)], score)
        (x, y), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)
        color = shader.get_color(int(cls))

        x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.rectangle(img, (x1, y1 - y - base),
                            (x1 + x, y1),
                            color, -1)
        img = cv2.putText(img,
                          msg,
                          (x1, y1),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255), 1)

        return img

    for bbox, score, cls in zip(bboxes, scores, classes):
        img = _draw_bbox(img, bbox, score, cls)

    return img
