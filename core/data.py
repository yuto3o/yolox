# -*- coding: utf-8 -*-
import numpy as np

from tensorflow.keras.utils import Sequence
from .utils import load_annotations, parse_annotation


class DataGenerator(Sequence):

    def __init__(self, mode, cfg):
        assert mode in ["train", "test"]
        self.cfg = cfg
        self.num_classes = cfg["core"]["num_classes"]
        self.mask = cfg["core"]["mask"]
        self.anchors = cfg["core"]["anchors"]
        self.max_boxes = cfg["core"]["max_boxes"]
        self.strides = cfg["core"]["strides"]

        self.path = self.cfg[mode]["annot_path"]
        self.annotations = load_annotations(self.path)
        self.num_samples = len(self.annotations)

        self.image_size = self.cfg[mode]["image_size"]
        self.batch_size = self.cfg[mode]["batch_size"]

        self.num_batch = self.num_samples // self.batch_size
        self._image_size = np.random.choice(self.image_size)

    def __len__(self):
        return self.num_batch

    def __getitem__(self, idx):

        self.grid_size = self._image_size // self.strides

        batch_image = np.zeros((self.batch_size, self._image_size, self._image_size, 3), dtype=np.float32)
        batch_label = [np.zeros((self.batch_size, size, size, len(mask_per_layer) * (5 + self.num_classes)),
                                dtype=np.float32)
                       for size, mask_per_layer in zip(self.grid_size, self.mask)]
        num = 0
        while num < self.batch_size:
            index = idx * self.batch_size + num

            if index >= self.num_samples:
                index -= self.num_samples

            annotation = self.annotations[index]
            # boxes: x1, y1, x2, y2, label
            image, bboxes = parse_annotation(annotation, (self._image_size, self._image_size))
            # TODO: image augment

            anchors = self.anchors / (self._image_size, self._image_size)
            bboxes_label = self.preprocess_true_boxes(bboxes, anchors)

            batch_image[num, :, :, :] = image

            for i in range(len(self.mask)):
                batch_label[i][num, :, :, :] = bboxes_label[i][:, :, :]

            num += 1

        return batch_image, tuple(batch_label)

    def on_epoch_end(self):
        np.random.shuffle(self.annotations)  # shuffle
        self._image_size = np.random.choice(self.image_size)  # other size

    def preprocess_true_boxes(self, bboxes, anchors):

        bboxes_label = [np.zeros((size, size, len(mask_per_layer), 5 + self.num_classes), np.float32)
                        for size, mask_per_layer in zip(self.grid_size, self.mask)]

        bboxes = np.asarray(bboxes)
        bboxes_class = bboxes[..., 4]
        bboxes = bboxes[..., :4]

        # smooth onehot
        onehot = np.eye(self.num_classes, dtype=np.float32)
        onehot = onehot[bboxes_class.astype(np.int)]
        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
        delta = 0.01
        smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

        # calculate anchor index for true boxes
        anchor_area = anchors[:, 0] * anchors[:, 1]
        bboxes_wh = bboxes[:, 2:4] - bboxes[:, 0:2]
        bboxes_wh = np.tile(np.expand_dims(bboxes_wh, 1),
                            (1, anchors.shape[0], 1))
        boxes_area = bboxes_wh[..., 0] * bboxes_wh[..., 1]
        intersection = np.minimum(bboxes_wh[..., 0], anchors[:, 0]) * np.minimum(bboxes_wh[..., 1],
                                                                                 anchors[:, 1])
        iou = intersection / (boxes_area + anchor_area - intersection)  # (N, A)
        anchor_idxs = np.argmax(iou, axis=-1)  # (N,)

        for i in range(len(bboxes)):
            search = np.where(self.mask == anchor_idxs[i])
            layer_idx = search[0][0]
            anchor_idx = search[1][0]

            coord_xy = (bboxes[i, 0:2] + bboxes[i, 2:4]) * 0.5
            coord_xy //= (1. / self.grid_size[layer_idx])
            coord_xy = coord_xy.astype(np.int)

            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, :4] = bboxes[i, :]
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 4:5] = 1.
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 5:] = smooth_onehot[i, :]

        return [layer.reshape([layer.shape[0], layer.shape[1], -1]) for layer in bboxes_label]


class SimpleGenerator:

    def __init__(self, mode, cfg):
        assert mode in ["train", "test"]
        self.mode = mode
        self.cfg = cfg
        self.num_classes = cfg["yolo"]["num_classes"]
        self.mask = cfg["yolo"]["mask"]
        self.anchors = cfg["yolo"]["anchors"]
        self.max_boxes = cfg["yolo"]["max_boxes"]
        self.strides = cfg["yolo"]["strides"]

        self.path = self.cfg[mode]["annot_path"]
        self.annotations = load_annotations(self.path)
        self.num_samples = len(self.annotations)

        self.image_size = self.cfg[mode]["image_size"]
        self.batch_size = self.cfg[mode]["batch_size"]
        self.num_batch = self.num_samples // self.batch_size

    def __call__(self):

        while True:
            batch_count = 0
            image_size = np.random.choice(self.image_size)
            self.grid_size = image_size // self.strides

            if self.mode == "train":
                np.random.shuffle(self.annotations)

            while batch_count < self.num_batch:
                batch_image = np.zeros((self.batch_size, image_size, image_size, 3), dtype=np.float32)
                batch_label = [np.zeros((self.batch_size, size, size, len(mask_per_layer) * (5 + self.num_classes)),
                                        dtype=np.float32)
                               for size, mask_per_layer in zip(self.grid_size, self.mask)]

                num = 0
                while num < self.batch_size:
                    index = batch_count * self.batch_size + num

                    if index >= self.num_samples:
                        index -= self.num_samples

                    annotation = self.annotations[index]
                    # boxes: x1, y1, x2, y2, label
                    image, bboxes = parse_annotation(annotation, (image_size, image_size))
                    # TODO: image augment
                    if self.mode == "train":
                        pass

                    anchors = self.anchors / (image_size, image_size)
                    bboxes_label = self.preprocess_true_boxes(bboxes, anchors)

                    batch_image[num, :, :, :] = image

                    for i in range(len(self.mask)):
                        batch_label[i][num, :, :, :] = bboxes_label[i][:, :, :]

                    num += 1

                batch_count += 1
                yield batch_image, tuple(batch_label)

    def preprocess_true_boxes(self, bboxes, anchors):

        bboxes_label = [np.zeros((size, size, len(mask_per_layer), 5 + self.num_classes), np.float32)
                        for size, mask_per_layer in zip(self.grid_size, self.mask)]

        bboxes = np.asarray(bboxes)
        bboxes_class = bboxes[..., 4]
        bboxes = bboxes[..., :4]

        # smooth onehot
        onehot = np.eye(self.num_classes, dtype=np.float32)
        onehot = onehot[bboxes_class.astype(np.int)]
        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
        delta = 0.01
        smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

        # calculate anchor index for true boxes
        anchor_area = anchors[:, 0] * anchors[:, 1]
        bboxes_wh = bboxes[:, 2:4] - bboxes[:, 0:2]
        bboxes_wh = np.tile(np.expand_dims(bboxes_wh, 1),
                            (1, anchors.shape[0], 1))
        boxes_area = bboxes_wh[..., 0] * bboxes_wh[..., 1]
        intersection = np.minimum(bboxes_wh[..., 0], anchors[:, 0]) * np.minimum(bboxes_wh[..., 1],
                                                                                 anchors[:, 1])
        iou = intersection / (boxes_area + anchor_area - intersection)  # (N, A)
        anchor_idxs = np.argmax(iou, axis=-1)  # (N,)

        for i in range(len(bboxes)):
            search = np.where(self.mask == anchor_idxs[i])
            layer_idx = search[0][0]
            anchor_idx = search[1][0]

            coord_xy = (bboxes[i, 0:2] + bboxes[i, 2:4]) * 0.5
            coord_xy //= (1. / self.grid_size[layer_idx])
            coord_xy = coord_xy.astype(np.int)

            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, :4] = bboxes[i, :]
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 4:5] = 1.
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 5:] = smooth_onehot[i, :]

        return [layer.reshape([layer.shape[0], layer.shape[1], -1]) for layer in bboxes_label]
