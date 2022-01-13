# -*- coding: utf-8 -*-
import numpy as np

from tensorflow.keras.utils import Sequence
from core.dataset import augment
from core.image import read_image, preprocess_image
from core.utils import decode_annotation, decode_name


class Dataset(Sequence):

    def __init__(self, cfg, verbose=0):

        self.verbose = verbose

        self.mask = cfg["yolo"]["mask"]
        self.anchors = cfg["yolo"]["anchors"]
        self.max_boxes = cfg["yolo"]["max_boxes"]
        self.strides = cfg["yolo"]["strides"]

        self.name_path = cfg['yolo']['name_path']
        self.anno_path = cfg["train"]["anno_path"]
        self.image_size = cfg["train"]["image_size"]
        self.batch_size = cfg["train"]["batch_size"]

        self.normal_method = cfg['train']["normal_method"]
        self.mosaic = cfg['train']['mosaic']
        self.label_smoothing = cfg['train']["label_smoothing"]

        self.annotation = decode_annotation(anno_path=self.anno_path)
        self.num_anno = len(self.annotation)
        self.name = decode_name(name_path=self.name_path)
        self.num_classes = len(self.name)

        # init
        self._image_size = np.random.choice(self.image_size)
        self._grid_size = self._image_size // self.strides

    def __len__(self):
        return int(np.ceil(float(len(self.annotation)) / self.batch_size))

    def __getitem__(self, idx):

        l_bound = idx * self.batch_size
        r_bound = (idx + 1) * self.batch_size

        if r_bound > len(self.annotation):
            r_bound = len(self.annotation)
            l_bound = r_bound - self.batch_size

        self._on_batch_start(idx)

        batch_image = np.zeros(
            (r_bound - l_bound, self._image_size, self._image_size, 3), dtype=np.float32)
        batch_label = [np.zeros((r_bound - l_bound, size, size, len(mask_per_layer) * (5 + self.num_classes)),
                                dtype=np.float32)
                       for size, mask_per_layer in zip(self._grid_size, self.mask)]

        for i, sub_idx in enumerate(range(l_bound, r_bound)):
            image, bboxes, labels = self._getitem(sub_idx)

            if self.mosaic:
                sub_idx = np.random.choice(
                    np.delete(np.arange(self.num_anno), idx), 3, False)
                image2, bboxes2, labels2 = self._getitem(sub_idx[0])
                image3, bboxes3, labels3 = self._getitem(sub_idx[1])
                image4, bboxes4, labels4 = self._getitem(sub_idx[2])
                image, bboxes, labels = augment.mosic(image, bboxes, labels,
                                                      image2, bboxes2, labels2,
                                                      image3, bboxes3, labels3,
                                                      image4, bboxes4, labels4)
            if self.normal_method:
                image = augment.random_distort(image)
                image = augment.random_grayscale(image)
                image, bboxes = augment.random_flip_lr(image, bboxes)
                image, bboxes = augment.random_rotate(image, bboxes, 15.)
                image, bboxes, labels = augment.random_crop_and_zoom(image, bboxes, labels,
                                                                     (self._image_size, self._image_size))

            image, bboxes, labels = augment.bbox_filter(image, bboxes, labels)
            labels = self._preprocess_true_boxes(bboxes, labels)

            batch_image[i] = image
            for j in range(len(self.mask)):
                batch_label[j][i, :, :, :] = labels[j][:, :, :]

        return batch_image, batch_label

    def _getitem(self, sub_idx):
        path, bboxes, labels = self.annotation[sub_idx]
        image = read_image(path)

        if len(bboxes) != 0:
            bboxes, labels = np.array(bboxes), np.array(labels)
        else:
            bboxes, labels = np.zeros((0, 4)), np.zeros((0,))

        image, bboxes = preprocess_image(
            image, (self._image_size, self._image_size), bboxes)
        labels = augment.onehot(labels, self.num_classes, self.label_smoothing)

        return image, bboxes, labels

    def _preprocess_true_boxes(self, bboxes, labels):

        bboxes_label = [np.zeros((size, size, len(mask_per_layer), 5 + self.num_classes), np.float32)
                        for size, mask_per_layer in zip(self._grid_size, self.mask)]

        bboxes = np.array(bboxes, dtype=np.float32)
        # calculate anchor index for true boxes
        anchor_area = self.anchors[:, 0] * self.anchors[:, 1]
        bboxes_wh = bboxes[:, 2:4] - bboxes[:, 0:2]

        bboxes_wh_exp = np.tile(np.expand_dims(
            bboxes_wh, 1), (1, self.anchors.shape[0], 1))
        boxes_area = bboxes_wh_exp[..., 0] * bboxes_wh_exp[..., 1]
        intersection = np.minimum(bboxes_wh_exp[..., 0], self.anchors[:, 0]) * np.minimum(bboxes_wh_exp[..., 1],
                                                                                          self.anchors[:, 1])
        iou = intersection / (boxes_area + anchor_area -
                              intersection + 1e-8)  # (N, A)
        best_anchor_idxs = np.argmax(iou, axis=-1)  # (N,)

        for i, bbox in enumerate(bboxes):

            search = np.where(self.mask == best_anchor_idxs[i])
            best_detect = search[0][0]
            best_anchor = search[1][0]

            coord_xy = (bbox[0:2] + bbox[2:4]) * 0.5
            coord_xy /= self.strides[best_detect]
            coord_xy = coord_xy.astype(np.int)

            bboxes_label[best_detect][coord_xy[1],
                                      coord_xy[0], best_anchor, :4] = bbox
            bboxes_label[best_detect][coord_xy[1],
                                      coord_xy[0], best_anchor, 4:5] = 1.
            bboxes_label[best_detect][coord_xy[1],
                                      coord_xy[0], best_anchor, 5:] = labels[i, :]

        return [layer.reshape([layer.shape[0], layer.shape[1], -1]) for layer in bboxes_label]

    def _on_batch_start(self, idx, patience=10):
        if idx % patience == 0:
            self._image_size = np.random.choice(self.image_size)
            self._grid_size = self._image_size // self.strides

            if self.verbose:
                print('Change image size to', self._image_size)

    def on_epoch_end(self):
        np.random.shuffle(self.annotation)  # shuffle
