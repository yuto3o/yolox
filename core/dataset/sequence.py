import numpy as np
import cv2

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

        self.anno_path = cfg["train"]["anno_path"]
        self.name_path = cfg['train']['name_path']
        self.image_size = cfg["train"]["image_size"]
        self.batch_size = cfg["train"]["batch_size"]

        self.normal_method = cfg['train']["normal_method"]
        self.mix_up = cfg['train']["mix_up"]
        self.cut_mix = cfg['train']['cut_mix']
        self.mosaic = cfg['train']['mosaic']
        self.label_smoothing = cfg['train']["label_smoothing"]

        self.annotation = decode_annotation(anno_path=self.anno_path)
        self.num_anno = len(self.annotation)
        self.name = decode_name(name_path=self.name_path)
        self.num_classes = len(self.name)

        # init
        self._image_size = np.random.choice(self.image_size)
        self._anchors = self.anchors / self._image_size
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

        batch_image = np.zeros((r_bound - l_bound, self._image_size, self._image_size, 3), dtype=np.float32)
        batch_label = [np.zeros((r_bound - l_bound, size, size, len(mask_per_layer) * (6 + self.num_classes)),
                                dtype=np.float32)
                       for size, mask_per_layer in zip(self._grid_size, self.mask)]

        for i, sub_idx in enumerate(range(l_bound, r_bound)):
            image, bboxes, labels = self._getitem(sub_idx)
            weights = np.full(len(labels), 1.)

            # high level augment

            if self.mix_up and np.random.randint(2):
                sub_idx2 = np.random.choice(np.delete(np.arange(self.num_anno), sub_idx))
                image2, bboxes2, labels2 = self._getitem(sub_idx2)
                image, bboxes, labels, weights = augment.mix_up(image, bboxes, labels, image2, bboxes2, labels2)

            elif self.cut_mix and np.random.randint(2):
                sub_idx2 = np.random.choice(np.delete(np.arange(self.num_anno), sub_idx))
                image2, bboxes2, labels2 = self._getitem(sub_idx2)
                image, bboxes, labels, weights = augment.cut_mix(image, bboxes, labels, image2, bboxes2, labels2)

            elif self.mosaic and np.random.randint(2):
                sub_idx = np.random.choice(np.delete(np.arange(self.num_anno), idx), 3, False)
                image2, bboxes2, labels2 = self._getitem(sub_idx[0])
                image3, bboxes3, labels3 = self._getitem(sub_idx[1])
                image4, bboxes4, labels4 = self._getitem(sub_idx[2])
                image, bboxes, labels, weights = augment.mosic(image, bboxes, labels,
                                                               image2, bboxes2, labels2,
                                                               image3, bboxes3, labels3,
                                                               image4, bboxes4, labels4)

            image, bboxes, labels, weights = augment.bbox_filter(image, bboxes, labels, weights)

            bboxes = np.divide(bboxes, self._image_size)
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0., 1.)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0., 1.)
            labels = self._preprocess_true_boxes(bboxes, labels, weights)

            batch_image[i] = image
            for j in range(len(self.mask)):
                batch_label[j][i, :, :, :] = labels[j][:, :, :]

        return batch_image, batch_label

    def _getitem(self, sub_idx):
        path, bboxes, labels = self.annotation[sub_idx]
        image = read_image(path)
        bboxes, labels = np.array(bboxes), np.array(labels)

        if self.normal_method:
            image = augment.random_distort(image)
            image = augment.random_grayscale(image)
            image, bboxes = augment.random_flip_lr(image, bboxes)
            image, bboxes = augment.random_rotate(image, bboxes)
            image, bboxes, labels = augment.random_crop_and_zoom(image, bboxes, labels,
                                                                 (self._image_size, self._image_size))
        else:

            image, bboxes = preprocess_image(image, (self._image_size, self._image_size), bboxes)

        labels = augment.onehot(labels, self.num_classes, self.label_smoothing)

        return image, bboxes, labels

    def _preprocess_true_boxes(self, bboxes, labels, weights):

        bboxes_label = [np.zeros((size, size, len(mask_per_layer), 6 + self.num_classes), np.float32)
                        for size, mask_per_layer in zip(self._grid_size, self.mask)]

        # !!! default mixup weight should be 1. not 0.
        for i in range(len(self.mask)):
            bboxes_label[i][:, :, :, 5:6] = 1.

        bboxes = np.array(bboxes, dtype=np.float32)
        # calculate anchor index for true boxes
        anchor_area = self._anchors[:, 0] * self._anchors[:, 1]
        bboxes_wh = bboxes[:, 2:4] - bboxes[:, 0:2]

        bboxes_wh_exp = np.tile(np.expand_dims(bboxes_wh, 1), (1, self._anchors.shape[0], 1))
        boxes_area = bboxes_wh_exp[..., 0] * bboxes_wh_exp[..., 1]
        intersection = np.minimum(bboxes_wh_exp[..., 0], self._anchors[:, 0]) * np.minimum(bboxes_wh_exp[..., 1],
                                                                                           self._anchors[:, 1])
        iou = intersection / (boxes_area + anchor_area - intersection + 1e-8)  # (N, A)
        anchor_idxs = np.argmax(iou, axis=-1)  # (N,)

        for i, bbox in enumerate(bboxes):
            search = np.where(self.mask == anchor_idxs[i])
            layer_idx = search[0][0]
            anchor_idx = search[1][0]

            coord_xy = (bbox[0:2] + bbox[2:4]) * 0.5
            coord_xy *= self._grid_size[layer_idx]
            coord_xy = coord_xy.astype(np.int)

            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, :4] = bbox
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 4:5] = 1.
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 5:6] = weights[i]
            bboxes_label[layer_idx][coord_xy[1], coord_xy[0], anchor_idx, 6:] = labels[i, :]

        return [layer.reshape([layer.shape[0], layer.shape[1], -1]) for layer in bboxes_label]

    def _on_batch_start(self, idx, patience=10):
        if idx % patience == 0:
            self._image_size = np.random.choice(self.image_size)
            self._grid_size = self._image_size // self.strides
            self._anchors = self.anchors / self._image_size

            if self.verbose:
                print('Change image size to', self._image_size)

    def on_epoch_end(self):
        np.random.shuffle(self.annotation)  # shuffle
