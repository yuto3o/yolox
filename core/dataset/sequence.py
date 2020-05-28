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


#########################
# Augment
#########################
# def onehot(labels, num_classes, smoothing):
#     bboxes_class = np.asarray(labels, dtype=np.int64)
#     labels = np.eye(num_classes, dtype=np.float32)
#     labels = labels[bboxes_class]
#
#     if smoothing:
#         uniform_distribution = np.full(num_classes, 1.0 / num_classes)
#         delta = 0.01
#         labels = labels * (1 - delta) + delta * uniform_distribution
#
#     return labels
#
#
# def random_distort(image, hue=18, saturation=1.5, exposure=1.5):
#     # determine scale factors
#     dhue = np.random.uniform(-hue, hue)
#     dsat = np.random.uniform(1. / saturation, saturation)
#     dexp = np.random.uniform(1. / exposure, exposure)
#
#     # convert RGB space to HSV space
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')
#
#     # change satuation and exposure
#     image[:, :, 1] *= dsat
#     image[:, :, 2] *= dexp
#
#     # change hue
#     image[:, :, 0] += dhue
#
#     image[:, :, 0] = np.clip(image[:, :, 0], 0., 179.)
#     image[:, :, 1] = np.clip(image[:, :, 1], 0., 255.)
#     image[:, :, 2] = np.clip(image[:, :, 2], 0., 255.)
#
#     # convert back to RGB from HSV
#     return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)
#
#
# def random_flip_lr(image, bboxes):
#     if np.random.randint(2):
#         h, w, _ = image.shape
#         image = image[:, ::-1, :]
#         bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
#
#     return image, bboxes
#
# def random_crop_and_zoom(image, bboxes, size, jitter=0.3):
#     net_w, net_h = size
#     h, w, _ = image.shape
#     dw = w * jitter
#     dh = h * jitter
#
#     rate = (w + np.random.uniform(-dw, dw)) / (h + np.random.uniform(-dh, dh))
#     scale = np.random.uniform(1/1.5, 1.5)
#
#     if (rate < 1):
#         new_h = int(scale * net_h)
#         new_w = int(new_h * rate)
#     else:
#         new_w = int(scale * net_w)
#         new_h = int(new_w / rate)
#
#     dx = int(np.random.uniform(0, net_w - new_w))
#     dy = int(np.random.uniform(0, net_h - new_h))
#
#     # image = cv2.resize(image, (new_w, new_h))
#     M = np.array([[new_w/w, 0., dx],
#                   [0., new_h/h, dy]], dtype=np.float32)
#     image = cv2.warpAffine(image, M, size, borderValue=(127, 127, 127))
#
#     # image = image[:net_h, :net_w, :]
#
#     bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * new_w / w + dx
#     bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * new_h / h + dy
#
#     bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, net_w)
#     bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, net_h)
#
#     filter_b = np.logical_or(bboxes[:, 0] >= bboxes[:, 2], bboxes[:, 1] >= bboxes[:, 3])
#     bboxes = bboxes[~filter_b]
#
#     return image, bboxes
#
# def _random_crop_and_zoom(image, bboxes, size, jitter=0.3):
#     net_w, net_h = size
#     h, w, _ = image.shape
#     dw = w * jitter
#     dh = h * jitter
#
#     rate = (w + np.random.uniform(-dw, dw)) / (h + np.random.uniform(-dh, dh))
#     scale = np.random.uniform(0.25, 2.)
#
#     if (rate < 1):
#         new_h = int(scale * net_h)
#         new_w = int(net_h * rate)
#     else:
#         new_w = int(scale * net_w)
#         new_h = int(net_w / rate)
#
#     dx = int(np.random.uniform(0, net_w - new_w))
#     dy = int(np.random.uniform(0, net_h - new_h))
#
#     image = cv2.resize(image, (new_w, new_h))
#
#     if dx > 0:
#         image = np.pad(image, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
#     else:
#         image = image[:, -dx:, :]
#
#     if (new_w + dx) < net_w:
#         image = np.pad(image, ((0, 0), (0, net_w - (new_w + dx)), (0, 0)), mode='constant', constant_values=127)
#
#     if dy > 0:
#         image = np.pad(image, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
#     else:
#         image = image[-dy:, :, :]
#     if (new_h + dy) < net_h:
#         image = np.pad(image, ((0, net_h - (new_h + dy)), (0, 0), (0, 0)), mode='constant', constant_values=127)
#
#     image = image[:net_h, :net_w, :]
#
#     bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * new_w / w + dx
#     bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * new_h / h + dy
#
#     bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, net_w)
#     bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, net_h)
#
#     filter_b = np.logical_or(bboxes[:, 0] >= bboxes[:, 2], bboxes[:, 1] >= bboxes[:, 3])
#     bboxes = bboxes[~filter_b]
#
#     return image, bboxes


#########################
# Utils
#########################
# def read_image(*args, **kwargs):
#     return cv2.cvtColor(cv2.imread(*args, **kwargs), cv2.COLOR_BGR2RGB)
#
#
# def decode_name(name_path):
#     with open(name_path, 'r') as f:
#         lines = f.readlines()
#     name = []
#     for line in lines:
#         line = line.strip()
#         if line:
#             name.append(line)
#     return name
#
#
# def decode_annotation(anno_path, type='y_true'):
#     with open(anno_path, 'r') as f:
#         lines = f.readlines()
#     anno = []
#     for line in lines:
#         line = line.strip()
#         if line:
#             anno.append(decode_line(line, type))
#     return anno
#
#
# def decode_line(line, type):
#     if type == 'y_true':
#         return decode_line_y_true(line)
#     elif type == 'y_pred':
#         return decode_line_y_pred(line)
#     else:
#         raise NotImplementedError(type)
#
#
# def decode_line_y_pred(line):
#     """
#     format
#     path x1,y1,x2,y2,label x1,y1,x2,y2,label...
#     :param line:
#     :return: {'path':str, 'bboxes':list, (x1,y1,x2,y2), 'labels':list }
#     """
#     items = line.split()
#     path = items[0]
#     items = items[1:]
#
#     bboxes = []
#     labels = []
#     confis = []
#     for item in items:
#         if not item:
#             continue
#         x1, y1, x2, y2, label, confi = item.split(',')
#         x1, y1, x2, y2, label, confi = float(x1), float(y1), float(x2), float(y2), int(label), float(confi)
#         bboxes.append([x1, y1, x2, y2])
#         labels.append(label)
#         confis.append(confi)
#
#     return path, bboxes, labels, confis
#
#
# def decode_line_y_true(line):
#     """
#     format
#     path x1,y1,x2,y2,label x1,y1,x2,y2,label...
#     :param line:
#     :return: {'path':str, 'bboxes':list, (x1,y1,x2,y2), 'labels':list }
#     """
#     items = line.split()
#     path = items[0]
#     items = items[1:]
#
#     bboxes = []
#     labels = []
#     for item in items:
#         if not item:
#             continue
#         x1, y1, x2, y2, label = item.split(',')
#         x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
#         bboxes.append([x1, y1, x2, y2])
#         labels.append(label)
#
#     return path, bboxes, labels
#
#
# import yaml
# import os
# import numpy as np
#
# from core.utils import decode_name
#
#
# def _decode_yaml_tuple(tuple_str):
#     return np.array(list(map(lambda x: list(map(int, str.split(x, ','))), tuple_str.split())))
#
#
# def decode_cfg(path):
#     print('Loading config from', path)
#     if not os.path.exists(path):
#         raise KeyError('%s does not exist ... ' % path)
#
#     with open(path, 'r') as f:
#         cfg = yaml.safe_load(f.read())
#
#         # some fields need to be decoded
#         cfg['yolo']['strides'] = list(map(int, cfg['yolo']['strides'].split(',')))
#         cfg['yolo']['mask'] = _decode_yaml_tuple(cfg['yolo']['mask'])
#         cfg['yolo']['anchors'] = _decode_yaml_tuple(cfg['yolo']['anchors'])
#
#         cfg['train']['image_size'] = list(map(int, cfg['train']['image_size'].split(',')))
#         cfg['train']['lr_base'] = float(cfg['train']['lr_base'])
#
#         cfg['test']['image_size'] = list(map(int, cfg['test']['image_size'].split(',')))
#
#     return cfg

# def _draw_bboxes_absolute(img, bboxes):
#     """
#     BBoxes is absolute Format
#     :param img: BGR, uint8
#     :param bboxes: x1, y1, x2, y2, int
#     :return: img, BGR, uint8
#     """
#
#     def _draw_bbox(img, bbox):
#         color = (255, 0, 0)
#
#         x1, y1, x2, y2 = bbox[:4]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
#         return img
#
#     for bbox in bboxes:
#         img = _draw_bbox(img, bbox)
#
#     return img

if __name__ == '__main__':
    cfg = decode_cfg("C:\\Users\\test\\Documents\\CodeHub\\Detection\\cfgs\\voc_yolov3_tiny.yaml")
    dataset = Dataset(cfg, verbose=1)

    for idx in range(len(dataset)):
        image_batch, batch_bboxes = dataset[idx]

        image = image_batch.astype(np.uint8)[..., ::-1]
        print(image.shape)
        image = _draw_bboxes_absolute(image.copy(), batch_bboxes)

        cv2.imshow('img', image)
        cv2.waitKey()
