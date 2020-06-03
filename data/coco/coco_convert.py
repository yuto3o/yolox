# -*- coding: utf-8 -*-
import os
import json
import pprint

from absl import logging, app, flags
from collections import defaultdict

flags.DEFINE_string('coco_path', None, 'path to coco dataset')
flags.DEFINE_string('name_path', None, 'path to coco name file')
flags.DEFINE_string('txt_output_path', None, 'path to output txt file')
flags.DEFINE_boolean('use_crowd', True, 'use crowd annotation')

FLAGS = flags.FLAGS


def convert(coco_path, coco_name_path, txt_output_path, use_crowd=True):
    def _read_txt_line(path):
        with open(path, 'r') as f:
            txt = f.readlines()

        return [line.strip() for line in txt]

    ann_path_train2017 = os.path.join(coco_path, 'annotations', 'instances_train2017.json')
    img_path_train2017 = os.path.join(coco_path, 'images', 'train2017')

    ann_path_val2017 = os.path.join(coco_path, 'annotations', 'instances_val2017.json')
    img_path_val2017 = os.path.join(coco_path, 'images', 'val2017')

    coco_name = _read_txt_line(coco_name_path)

    def _check_bbox(x1, y1, x2, y2, w, h):

        if x1 < 0 or x2 < 0 or x1 > w or x2 > w or y1 < 0 or y2 < 0 or y1 > h or y2 > h:
            logging.warning('cross boundary (' + str(w) + ',' + str(h) + '),(' + ','.join(
                [str(x1), str(y1), str(x2), str(y2)]) + ')')

            return str(min(max(x1, 0.), w)), str(min(max(y1, 0.), h)), str(min(max(x2, 0.), w)), str(
                min(max(y2, 0.), h))

        return x1, y1, x2, y2

    def _write_to_text(ann_path, img_path, txt_path):
        dataset = json.load(open(ann_path, 'r'))
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns = defaultdict(list)
        if 'annotations' in dataset:
            for ann in dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in dataset:
            for img in dataset['images']:
                imgs[img['id']] = img

        if 'categories' in dataset:
            for cat in dataset['categories']:
                cats[cat['id']] = cat

        print('Categories')
        pprint.pprint(cats)
        print('index created!')

        with open(txt_path, 'w') as f:
            for img_id, img in imgs.items():
                anns = imgToAnns[img_id]
                iw, ih = img['width'], img['height']
                file_name = img['file_name']

                line = os.path.join(img_path, file_name)

                for ann in anns:

                    label = cats[ann['category_id']]['name']
                    if label not in coco_name:
                        continue
                    if not use_crowd and ann['iscrowd'] == 1:
                        continue
                    idx = coco_name.index(label)
                    x, y, w, h = ann['bbox']
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    x1, y1, x2, y2 = _check_bbox(x1, y1, x2, y2, iw, ih)

                    line += ' ' + ','.join([str(x1), str(y1), str(x2), str(y2), str(idx)])

                logging.info(line)
                f.write(line + '\n')

    _write_to_text(ann_path_train2017, img_path_train2017, os.path.join(txt_output_path, 'train2017.txt'))
    _write_to_text(ann_path_val2017, img_path_val2017, os.path.join(txt_output_path, 'val2017.txt'))


def main(_argv):
    convert(FLAGS.coco_path, FLAGS.name_path, FLAGS.txt_output_path, FLAGS.use_crowd)


if __name__ == '__main__':
    app.run(main)
