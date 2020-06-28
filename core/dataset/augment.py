# -*- coding: utf-8 -*-
import numpy as np
import cv2


def mosic(image, bboxes, labels,
          image2, bboxes2, labels2,
          image3, bboxes3, labels3,
          image4, bboxes4, labels4,
          min_offset=0.3):
    h, w = image.shape[0], image.shape[1]

    mix_img = np.zeros(shape=(h, w, 3), dtype='uint8')

    cut_x = np.random.randint(w * min_offset, w * (1 - min_offset))
    cut_y = np.random.randint(h * min_offset, h * (1 - min_offset))

    # s = (cut_x * cut_y) / (w * h)
    # s2 = ((w - cut_x) * cut_y) / (w * h)
    # s3 = (cut_x * (h - cut_y)) / (w * h)
    # s4 = ((w - cut_x) * (h - cut_y)) / (w * h)

    mix_img[:cut_y, :cut_x] = image[:cut_y, :cut_x]
    mix_img[:cut_y, cut_x:] = image2[:cut_y, cut_x:]
    mix_img[cut_y:, :cut_x] = image3[cut_y:, :cut_x]
    mix_img[cut_y:, cut_x:] = image4[cut_y:, cut_x:]

    keep_idx, bboxes = clip_bbox(bboxes, (0, 0, cut_x, cut_y))
    keep_idx2, bboxes2 = clip_bbox(bboxes2, (cut_x, 0, w, cut_y))
    keep_idx3, bboxes3 = clip_bbox(bboxes3, (0, cut_y, cut_x, h))
    keep_idx4, bboxes4 = clip_bbox(bboxes4, (cut_x, cut_y, w, h))

    mix_bboxes = np.vstack((bboxes, bboxes2, bboxes3, bboxes4))
    mix_labels = np.vstack((labels[keep_idx], labels2[keep_idx2], labels3[keep_idx3], labels4[keep_idx4]))

    return mix_img, mix_bboxes, mix_labels


def clip_bbox(bboxes, target_bbox):
    tx1, ty1, tx2, ty2 = target_bbox

    x1 = np.maximum(bboxes[..., 0], tx1)
    y1 = np.maximum(bboxes[..., 1], ty1)
    x2 = np.minimum(bboxes[..., 2], tx2)
    y2 = np.minimum(bboxes[..., 3], ty2)

    new_bbox = np.stack([x1, y1, x2, y2], axis=-1)
    v_ioa = ioa(new_bbox, bboxes)
    keep_idx = v_ioa > 0.2

    return keep_idx, new_bbox[keep_idx]


def ioa(bboxes, target_bboxes):
    w = np.maximum(bboxes[..., 2] - bboxes[..., 0], 0)
    h = np.maximum(bboxes[..., 3] - bboxes[..., 1], 0)

    tw = np.maximum(target_bboxes[..., 2] - target_bboxes[..., 0], 0)
    th = np.maximum(target_bboxes[..., 3] - target_bboxes[..., 1], 0)

    ioa = w * h / np.maximum(tw * th, 1e-8)

    return ioa


# def keep_bbox_within(bboxes, target_bbox):
#     tx1, ty1, tx2, ty2 = target_bbox
#
#     if not isinstance(bboxes, np.ndarray):
#         bboxes = np.asarray(bboxes)
#
#     x1 = np.maximum(bboxes[..., 0], tx1)
#     y1 = np.maximum(bboxes[..., 1], ty1)
#     x2 = np.minimum(bboxes[..., 2], tx2)
#     y2 = np.minimum(bboxes[..., 3], ty2)
#
#     int_w = np.maximum(x2 - x1, 0)
#     int_h = np.maximum(y2 - y1, 0)
#     int_area = int_w * int_h
#
#     bboxes = np.stack([x1, y1, x2, y2], axis=-1)
#     # keep_idx = np.any(np.not_equal(bboxes, 0), axis=-1)
#     keep_idx = int_area > 0
#
#     return keep_idx, bboxes[keep_idx]


# def cut_mix(image, bboxes, labels, image2, bboxes2, labels2, beta=1):
#     """
#     CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
#     """
#
#     def rand_bbox(W, H, lambd):
#         cut_rat = np.sqrt(1. - lambd)
#         cut_w = np.int(W * cut_rat)
#         cut_h = np.int(H * cut_rat)
#
#         # uniform
#         x1 = np.random.randint(0, W - cut_w)
#         y1 = np.random.randint(0, H - cut_h)
#         x2 = x1 + cut_w
#         y2 = y1 + cut_h
#
#         return x1, y1, x2, y2
#
#     H, W = image.shape[0], image.shape[1]
#     lambd = np.random.beta(beta, beta)
#     min, max = 0.3, 0.8
#     lambd = min + (max - min) * lambd
#
#     x1, y1, x2, y2 = rand_bbox(W, H, lambd)
#     mix_img = image.copy()
#     mix_img[x1:x2, y1:y2] = image2[x1:x2, y1:y2]
#     # adjust lambda to exactly match pixel ratio
#     lambd = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
#
#     mix_bboxes = np.vstack((bboxes, bboxes2))
#     mix_labels = np.vstack((labels, labels2))
#     mix_weights = np.hstack((np.full(len(labels), lambd),
#                              np.full(len(labels2), (1. - lambd))))
#
#     return mix_img, mix_bboxes, mix_labels, mix_weights


def mix_up(image, bboxes, labels, image2, bboxes2, labels2, alpha=None, beta=None):
    if alpha is None or beta is None:
        # Yolo use fixed 0.5
        lambd = 0.5
    else:
        lambd = np.random.beta(beta, beta)

    H = max(image.shape[0], image2.shape[0])
    W = max(image.shape[1], image2.shape[1])
    mix_img = np.zeros(shape=(H, W, 3), dtype='float32')
    mix_img[:image.shape[0], :image.shape[1], :] = image.astype('float32') * lambd
    mix_img[:image2.shape[0], :image2.shape[1], :] += image2.astype('float32') * (1. - lambd)
    mix_img = mix_img.astype(np.uint8)

    mix_bboxes = np.vstack((bboxes, bboxes2))
    mix_labels = np.vstack((labels, labels2))
    mix_weights = np.hstack((np.full(len(labels), lambd),
                             np.full(len(labels2), (1. - lambd))))

    return mix_img, mix_bboxes, mix_labels, mix_weights


def onehot(labels, num_classes, smoothing):
    bboxes_class = np.asarray(labels, dtype=np.int64)
    labels = np.eye(num_classes, dtype=np.float32)
    labels = labels[bboxes_class]

    if smoothing:
        uniform_distribution = np.full(num_classes, 1.0 / num_classes)
        delta = 0.1
        labels = labels * (1 - delta) + uniform_distribution * delta

    return labels


def random_grayscale(image, alpha=(0.0, 1.0)):
    alpha = alpha[0] + np.random.uniform() * (alpha[1] - alpha[0])

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_gray = np.expand_dims(img_gray, axis=-1)
    img_gray = np.tile(img_gray, (1, 1, 3))

    img_gray = img_gray.astype(np.float32)
    img = image.astype(np.float32)
    img = img + alpha * (img_gray - img)
    img = np.clip(img, 0., 255.)
    image = img.astype(np.uint8)

    return image


def random_distort(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = np.random.uniform(1. / saturation, saturation)
    dexp = np.random.uniform(1. / exposure, exposure)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue

    image[:, :, 0] = np.clip(image[:, :, 0], 0., 179.)
    image[:, :, 1] = np.clip(image[:, :, 1], 0., 255.)
    image[:, :, 2] = np.clip(image[:, :, 2], 0., 255.)

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def random_rotate(image, bboxes, angle=7.):
    angle = np.random.uniform(-angle, angle)

    h, w, _ = image.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, m, (w, h), borderValue=(127, 127, 127))

    if len(bboxes) != 0:
        top_left = bboxes[..., [0, 1]]
        top_right = bboxes[..., [2, 1]]
        bottom_left = bboxes[..., [0, 3]]
        bottom_right = bboxes[..., [2, 3]]

        # N, 4, 2
        points = np.stack([top_left, top_right, bottom_left, bottom_right], axis=-2)
        points_3d = np.ones(points.shape[:-1] + (3,), np.float32)
        points_3d[..., :2] = points

        # points = m @ points_3d[0].T
        points = map(lambda x: m @ x.T, points_3d)
        points = np.array(list(points))
        points = np.transpose(points, [0, 2, 1])

        bboxes[..., 0] = np.min(points[..., 0], axis=-1)
        bboxes[..., 1] = np.min(points[..., 1], axis=-1)
        bboxes[..., 2] = np.max(points[..., 0], axis=-1)
        bboxes[..., 3] = np.max(points[..., 1], axis=-1)

        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)

    return image, bboxes


def random_flip_lr(image, bboxes):
    if np.random.randint(2):
        h, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes


def random_crop_and_zoom(image, bboxes, labels, size, jitter=0.3):
    net_w, net_h = size
    h, w, _ = image.shape
    dw = w * jitter
    dh = h * jitter

    rate = (w + np.random.uniform(-dw, dw)) / (h + np.random.uniform(-dh, dh))
    scale = np.random.uniform(1 / 1.5, 1.5)

    if (rate < 1):
        new_h = int(scale * net_h)
        new_w = int(new_h * rate)
    else:
        new_w = int(scale * net_w)
        new_h = int(new_w / rate)

    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))

    M = np.array([[new_w / w, 0., dx],
                  [0., new_h / h, dy]], dtype=np.float32)
    image = cv2.warpAffine(image, M, size, borderValue=(127, 127, 127))

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * new_w / w + dx
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * new_h / h + dy

    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, net_w)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, net_h)

    filter_b = np.logical_or(bboxes[:, 0] >= bboxes[:, 2], bboxes[:, 1] >= bboxes[:, 3])
    bboxes = bboxes[~filter_b]
    labels = labels[~filter_b]


    return image, bboxes, labels


def bbox_filter(image, bboxes, labels):
    """
    Maginot Line
    """

    h, w, _ = image.shape

    x1 = np.maximum(bboxes[..., 0], 0.)
    y1 = np.maximum(bboxes[..., 1], 0.)
    x2 = np.minimum(bboxes[..., 2], w - 1e-8)
    y2 = np.minimum(bboxes[..., 3], h - 1e-8)

    int_w = np.maximum(x2 - x1, 0)
    int_h = np.maximum(y2 - y1, 0)
    int_area = int_w * int_h

    bboxes = np.stack([x1, y1, x2, y2], axis=-1)
    # keep_idx = np.any(np.not_equal(bboxes, 0), axis=-1)
    keep_idx = int_area > 0.
    return image, bboxes[keep_idx], labels[keep_idx]
