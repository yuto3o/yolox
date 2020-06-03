# -*- coding: utf-8 -*-
import numpy as np
import cv2

from core.utils import decode_annotation

# Parameters
K = 6
image_size = 416
dataset_path = './data/pascal_voc/train.txt'

print('Num of Clusters is', K)
print('Base Image Size is', image_size)

# Read Dataset
anns = decode_annotation(dataset_path)


def resize_bboxes(path, bboxes):
    image = cv2.imread(path)
    h, w, _ = image.shape

    scale = min(image_size / w, image_size / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (image_size - nw) // 2, (image_size - nh) // 2

    bboxes = np.asarray(bboxes).astype(np.float32)
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, image_size - 1)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, image_size - 1)
    bboxes = bboxes.astype(np.int)
    return bboxes


bboxes = [resize_bboxes(ann[0], ann[1]) for ann in anns]

# Accumulate bboxes
bboxes = np.concatenate(bboxes, axis=0)
w = bboxes[:, 2] - bboxes[:, 0]
h = bboxes[:, 3] - bboxes[:, 1]
bboxes = np.stack((w, h), axis=-1)

N = len(bboxes)
clusters = bboxes[np.random.choice(N, K, replace=False)]

print('Dataset has', N, 'annotations')


def iou(lbbox, rbbox):
    lbbox = np.expand_dims(lbbox, axis=1)
    rbbox = np.expand_dims(rbbox, axis=0)

    lbbox_area = lbbox[..., 0] * lbbox[..., 1]
    rbbox_area = rbbox[..., 0] * rbbox[..., 1]

    intersection = np.minimum(lbbox[..., 0], rbbox[..., 0]) * np.minimum(lbbox[..., 1], rbbox[..., 1])
    iou = intersection / (lbbox_area + rbbox_area - intersection + 1e-8)  # (M, N)
    return iou


assign = np.zeros((N,))
while True:

    distances = 1 - iou(bboxes, clusters)

    _assign = np.argmin(distances, axis=1)
    if (assign == _assign).all():
        break  # clusters won't change
    for k in range(K):
        clusters[k] = np.median(bboxes[_assign == k], axis=0)

    assign = _assign

clusters = sorted(clusters, key=lambda x: x[0])

format = "{},{}"
info = "{},{}".format(clusters[0][0], clusters[0][1])
for w, h in clusters[1:]:
    info += ' ' + format.format(w, h)

print('K-Means Result:')
print(info)
