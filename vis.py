# -*- coding: utf-8 -*-
import numpy as np
import cv2

COLOR_MAP = (
 (41, 36, 33),
 (192, 192, 192),
 (128, 138, 135),
 (112, 128, 105),
 (128, 128, 105),
 (250, 235, 215),
 (240, 255, 255),
 (255, 235, 205),
 (252, 230, 201),
 (220, 220, 220),
 (240, 255, 240),
 (250, 240, 230),
 (255, 222, 173),
 (156, 102, 31),
 (227, 23, 13),
 (255, 127, 80),
 (178, 34, 34),
 (176, 48, 96),
 (255, 192, 203),
 (135, 38, 87),
 (250, 128, 114),
 (255, 99, 71),
 (255, 69, 0),
 (255, 0, 255),
 (176, 224, 230),
 (65, 105, 225),
 (106, 90, 205),
 (135, 206, 235),
 (0, 255, 255),
 (56, 94, 15),
 (8, 46, 84),
 (127, 255, 212),
 (64, 224, 208),
 (0, 255, 0),
 (61, 145, 64),
 (0, 201, 87),
 (124, 252, 0),
 (50, 205, 50),
 (189, 252, 201),
 (107, 142, 35),
 (48, 128, 20),
 (46, 139, 87),
 (0, 255, 127),
 (160, 32, 240),
 (138, 43, 226),
 (160, 102, 211),
 (218, 112, 214),
 (221, 160, 221),
 (255, 255, 0),
 (227, 207, 87),
 (255, 153, 18),
 (235, 142, 85),
 (255, 227, 132),
 (255, 215, 0),
 (218, 165, 105),
 (227, 168, 105),
 (255, 97, 0),
 (237, 145, 33),
 (255, 128, 0),
 (245, 222, 179),
 (128, 42, 42),
 (163, 148, 128),
 (138, 54, 15),
 (210, 105, 30),
 (255, 125, 64),
 (240, 230, 140),
 (188, 143, 143),
 (115, 74, 18),
 (94, 38, 18),
 (160, 82, 45),
 (244, 164, 96),
 (210, 180, 140),
 (0, 0, 255),
 (61, 89, 171),
 (30, 144, 255),
 (11, 23, 70),
 (3, 168, 158),
 (25, 25, 112),
 (51, 161, 201),
 (0, 199, 140))

def plot(imgs, bboxes, scores, labels, coco_name, output_path=None):
  for img, bbox, score, label in zip(imgs, bboxes, scores, labels):

    img = img * 255.
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)

    bbox = bbox[score>0]
    label = label[score>0]
    score = score[score>0]
    for _bbox, _label, _score in zip(bbox, label, score):

      _label = int(_label)
      _bbox = tuple(_bbox.astype(np.int))
      print(coco_name[_label], ':', _score)
      print('--- bbox:', _bbox)

      cv2.rectangle(img, _bbox[:2], _bbox[2:4], COLOR_MAP[_label], 5)
      cv2.rectangle(img, _bbox[:2],
                    (_bbox[0]+9*(8+len(coco_name[_label])), _bbox[1]+15),
                    COLOR_MAP[_label], -1)
      cv2.putText(img,
                  '{} {:.2f}%'.format(coco_name[_label], _score * 100),
                  (_bbox[0], _bbox[1]+10),
                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    if output_path:
      cv2.imwrite(output_path, img);

    cv2.imshow('detection', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
























