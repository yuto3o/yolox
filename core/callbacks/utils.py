# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import sys

from core.image import preprocess_image, postprocess_image, read_image


def keras_bar(i, nums, width=30):
    numdigits = int(np.log10(nums)) + 1
    bar = ('%' + str(numdigits) + 'd/%d [') % (i, nums)
    prog = float(i) / nums
    prog_width = int(width * prog)
    if prog_width > 0:
        bar += ('=' * (prog_width - 1))
        if i < nums:
            bar += '>'
        else:
            bar += '='
    bar += ('.' * (width - prog_width))
    bar += ']'
    return bar


def local_eval(func, model, image_size, test_path, name_path, verbose):
    tmp_path = os.path.join('tmp' + time.strftime("%Y%m%d%H%M", time.localtime()))
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with open(test_path) as f:
        lines = f.readlines()

    paths = [line.split()[0] for line in lines]

    infer_time = []
    with open(tmp_path, 'a+') as f:
        for i, path in enumerate(paths, 1):
            if i == 1:
                sys.stdout.write('\n')
            sys.stdout.write('\r' + keras_bar(i, len(paths)))
            image = read_image(path)
            h, w = image.shape[:2]
            image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
            images = np.expand_dims(image, axis=0)

            tic = time.time()
            bboxes, scores, classes, valid_detections = model.predict(images)
            toc = time.time()
            infer_time.append(toc - tic)

            bboxes = bboxes[0][:valid_detections[0]]
            scores = scores[0][:valid_detections[0]]
            classes = classes[0][:valid_detections[0]]

            _, bboxes = postprocess_image(image, (w, h), bboxes)

            line = path
            for bbox, score, cls in zip(bboxes, scores, classes):
                x1, y1, x2, y2 = bbox
                line += " {:.2f},{:.2f},{:.2f},{:.2f},{},{:.4f}".format(x1, y1, x2, y2, int(cls), score)

            f.write(line + '\n')

    ans = func(test_path, tmp_path, name_path, verbose=verbose)
    # remove tmp
    os.remove(tmp_path)

    if verbose:
        if len(infer_time) > 5:
            s = np.mean(infer_time[5:])
        else:
            s = np.mean(infer_time)

        print('\nInference time', s*1000, 'ms')

    return ans
