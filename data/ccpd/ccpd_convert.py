# -*- coding: utf-8 -*-
import os
import cv2

from absl import logging, app, flags

flags.DEFINE_string('ccpd_path', None, 'path to ccpd dataset')
flags.DEFINE_string('name_path', None, 'path to coco name file')
flags.DEFINE_string('txt_output_path', None, 'path to output txt file')

FLAGS = flags.FLAGS


def convert(ccpd_path, ccpd_name_path, txt_output_path):

    def _read_txt_line(path):
        with open(path, 'r') as f:
            txt = f.readlines()

        return [line.strip() for line in txt]

    ccpd_name = _read_txt_line(ccpd_name_path)

    def _check_bbox(x1, y1, x2, y2, w, h):

        if x1 < 0 or x2 < 0 or x1 > w or x2 > w or y1 < 0 or y2 < 0 or y1 > h or y2 > h:
            logging.warning('cross boundary (' + str(w) + ',' + str(h) + '),(' + ','.join(
                [str(x1), str(y1), str(x2), str(y2)]) + ')')

            return str(min(max(x1, 0.), w)), str(min(max(y1, 0.), h)), str(min(max(x2, 0.), w)), str(
                min(max(y2, 0.), h))

        return x1, y1, x2, y2

    def _write_to_text(img_path, txt_path):

        with open(img_path, 'r') as f:
            img_names = f.readlines()

        with open(txt_path, 'w') as f:
            for img_name in img_names:
                img_name = img_name.strip()

                img_path = os.path.join(ccpd_path, *img_name.split('/'))

                img = cv2.imread(img_path)
                h, w, _ = img.shape

                line = img_path

                iname = os.path.basename(img_name).rsplit('.', 1)[0].split('-')
                [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
                [x1, y1], [x2, y2] = leftUp, rightDown
                idx = ccpd_name.index('license plate')

                x1, y1, x2, y2 = _check_bbox(x1, y1, x2, y2, w, h)
                line += ' ' + ','.join([str(x1), str(y1), str(x2), str(y2), str(idx)])

                logging.info(line)
                f.write(line + '\n')

    img_path_train = os.path.join(ccpd_path, 'splits', 'train.txt')
    img_path_test = os.path.join(ccpd_path, 'splits', 'test.txt')
    img_path_val = os.path.join(ccpd_path, 'splits', 'val.txt')

    img_path_blur = os.path.join(ccpd_path, 'splits', 'ccpd_blur.txt')
    img_path_challenge = os.path.join(ccpd_path, 'splits', 'ccpd_challenge.txt')
    img_path_db = os.path.join(ccpd_path, 'splits', 'ccpd_db.txt')
    img_path_fn = os.path.join(ccpd_path, 'splits', 'ccpd_fn.txt')
    img_path_rotate = os.path.join(ccpd_path, 'splits', 'ccpd_rotate.txt')
    img_path_tilt = os.path.join(ccpd_path, 'splits', 'ccpd_tilt.txt')

    _write_to_text(img_path_train, os.path.join(txt_output_path, 'train.txt'))
    _write_to_text(img_path_test, os.path.join(txt_output_path, 'test.txt'))
    _write_to_text(img_path_val, os.path.join(txt_output_path, 'val.txt'))

    _write_to_text(img_path_blur, os.path.join(txt_output_path, 'blur.txt'))
    _write_to_text(img_path_challenge, os.path.join(txt_output_path, 'challenge.txt'))
    _write_to_text(img_path_db, os.path.join(txt_output_path, 'db.txt'))
    _write_to_text(img_path_fn, os.path.join(txt_output_path, 'fn.txt'))
    _write_to_text(img_path_rotate, os.path.join(txt_output_path, 'rotate.txt'))
    _write_to_text(img_path_tilt, os.path.join(txt_output_path, 'tilt.txt'))


def main(_argv):
    convert(FLAGS.ccpd_path, FLAGS.name_path, FLAGS.txt_output_path)


if __name__ == '__main__':
    app.run(main)
