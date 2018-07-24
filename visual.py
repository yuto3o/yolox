# -*- coding: utf-8 -*-
import cv2
import numpy as np

from itertools import product

PIXEL_VALUE = [50, 90, 140, 180, 220]
np.random.shuffle(PIXEL_VALUE)
COLOR_MAP = tuple(product(PIXEL_VALUE, PIXEL_VALUE, PIXEL_VALUE))[:80]

def vis(imgs, predictions, input_size, coco_name, output=None):
  """Visualize

  Params:
    imgs: list, consists of PIL.Image object
    inputs_sizes: tuple, (w, h)
  """

  for img, prediction in zip(imgs, predictions):
    for label, bboxes in prediction.items():
      for bbox, confidence in bboxes:
        bbox = descale(img.shape[-2::-1], input_size, bbox)
        print(coco_name[label], ':', confidence)
        print('--- bbox:', bbox)
        cv2.rectangle(img, bbox[:2], bbox[2:4], COLOR_MAP[label], 5)
        cv2.rectangle(img, bbox[:2],
                      (bbox[0]+9*(8+len(coco_name[label])), bbox[1]+15),
                      COLOR_MAP[label], -1)
        cv2.putText(img,
                    '{} {:.2f}%'.format(coco_name[label], confidence * 100),
                    (bbox[0], bbox[1]+10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255),1)

    if output:
      cv2.imwrite(output, img);

    cv2.imshow('detection', img)
    cv2.waitKey()
    cv2.destroyAllWindows()



def descale(orignal_size, input_size, bbox):

  scale = np.divide(orignal_size, input_size)
  bbox = np.reshape((np.reshape(bbox,[2,2])*scale),-1)
  print(scale)
  print(bbox)
  return tuple(bbox.astype(np.int))


