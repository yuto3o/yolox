import numpy as np
import cv2
import os


def decode_yaml_tuple(tuple_str):
    return np.array(list(map(lambda x: list(map(int, str.split(x, ','))), tuple_str.split())))


def load_annotations(path):
    if not os.path.exists(path):
        raise KeyError("%s does not exist ... " % path)
    with open(path, 'r') as f:
        annotations = f.readlines()
        annotations = [annotation.strip() for annotation in annotations if annotation]

    return annotations


def load_names(path):
    if not os.path.exists(path):
        raise KeyError("%s does not exist ... " % path)
    coco = {}
    with open(path, 'rt') as file:
        for index, label in enumerate(file):
            if label:
                coco[index] = label.strip()

    return coco


def parse_annotation(annotation, size):
    line = annotation.split()
    image_path = line[0]

    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)

    image = cv2.imread(image_path)
    bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

    image, bboxes = preprocess_image(size, image, bboxes)

    return image, bboxes


def preprocess_image(size, image_path, bboxes=None):
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB).astype(np.float32)

    iw, ih = size, size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=127.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if bboxes is None:
        return image_paded

    else:
        bboxes = bboxes.astype(np.float32)
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] * scale + dw) / iw
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] * scale + dh) / ih

        return image_paded, bboxes

