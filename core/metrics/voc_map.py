# -*- coding: utf-8 -*-
import os
import numpy as np

from core.utils import decode_annotation, decode_name


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def VOCEval(y_true_path,
            y_pred_path,
            name_path,
            ovthresh=0.5,
            use_07_metric=False,
            verbose=0):
    """
    :param y_true_path:
    :param y_pred_path:
    :param ovthresh: Overlap threshold (default = 0.5)
    :param use_07_metric: Whether to use VOC07's 11 point AP computation (default False)
    :return:
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # load y_true
    y_true = decode_annotation(y_true_path, type='y_true')
    # load y_pred
    y_pred = decode_annotation(y_pred_path, type='y_pred')

    names = decode_name(name_path)
    ans = {}

    for classname_idx in range(len(names)):

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imgs_path, bboxes, labels in y_true:
            image_idx = os.path.basename(imgs_path).split('.')[0]
            bbox = [bbox for bbox, label in zip(bboxes, labels) if label == classname_idx]
            bbox = np.array(bbox)
            det = [False] * len(bbox)
            diff = [False] * len(bbox)
            npos += len(bbox)
            class_recs[image_idx] = {'bbox': bbox,
                                     'det': det,
                                     'difficult': diff}
        # extract pd objects for this class
        image_ids = []
        BB = []
        confidence = []
        for imgs_path, bboxes, labels, confis in y_pred:
            image_idx = os.path.basename(imgs_path).split('.')[0]

            for bbox, label, confi in zip(bboxes, labels, confis):
                if label != classname_idx:
                    continue

                image_ids.append(image_idx)
                BB.append(bbox)
                confidence.append(confi)

        image_ids = np.array(image_ids)
        BB = np.array(BB)
        confidence = np.array(confidence)

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) * (BBGT[:, 3] - BBGT[:, 1] + 1.)
                       - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        ans[names[classname_idx]] = {'rec': rec, 'prec': prec, 'ap': ap}

    if verbose > 0:
        print("\nOn Test Data")
        print("class          |AP@50")

    mAP = []
    for key in ans.keys():
        ap = ans[key]['ap']
        mAP.append(ap)

        if verbose > 0:
            print("{:>15}|{:>15.2%}".format(key, ap))

    mAP = np.mean(mAP)
    if verbose > 0:
        print("{:>15}|{:>15.2%}".format('mAP', mAP))

    return mAP
