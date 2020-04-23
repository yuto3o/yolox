# -*- coding: utf-8 -*-
from absl import app, flags
from collections import defaultdict

import numpy as np
import os

# VOC mAP
# y_true file:
#     path, (x1, y1, x2, y2, cls_idx), ...
# y_pred file:
#     path, (x1, y1, x2, y2, cls_idx, confidence), ...

flags.DEFINE_string("gt_path", "../disc/annotation.txt", "path to Ground Truth: path, (x1, y1, x2, y2, cls_idx), ...")
flags.DEFINE_string("pd_path", "../disc/prediction.txt", "path to Predictation: path, (x1, y1, x2, y2, cls_idx, confidence), ...")
flags.DEFINE_string("cls_path", "../disc/coco.name", "path to the file(list of classes)")

FLAGS = flags.FLAGS

MINIOVERLAP = 0.5

def main(_argv):

    gt_path = FLAGS.gt_path
    pd_path = FLAGS.pd_path
    cls_path = FLAGS.cls_path

    gt, pd = read_file(gt_path, pd_path)

    classes, num_classes = load_names(cls_path)

    rec, prec, ap = report(gt, pd, num_classes)

    print(rec, prec, ap)



def read_file(gt_path, pd_path):

    if not (gt_path and os.path.exists(gt_path)):
        raise FileNotFoundError("file is not existed: %s" % gt_path)

    with open(gt_path, "r") as f:

        GT = {}
        for line in f.readlines():
            line = line.strip().split()
            path = line[0]
            GT[path] = defaultdict(list)

            if len(line) == 1:
                continue

            for info in line[1:]:
                x1, y1, x2, y2, cls_idx = info.split(",")
                GT[path][int(cls_idx)].append({"bbox": (float(x1), float(y1), float(x2), float(y2)),
                                               "hit": False})

    if not (pd_path and os.path.exists(pd_path)):
        raise FileNotFoundError("file is not existed: %s" % pd_path)

    with open(pd_path, "r") as f:
        PD = {}
        for line in f.readlines():
            line = line.strip().split()
            path = line[0]
            PD[path] = defaultdict(list)

            if len(line) == 1:
                continue

            for info in line[1:]:
                x1, y1, x2, y2, cls_idx, confi = info.split(",")
                PD[path][int(cls_idx)].append({"bbox": (float(x1), float(y1), float(x2), float(y2)),
                                               "confi": float(confi)})

        if GT.keys() != PD.keys():
            raise LookupError("File Path Not Match !")

        return GT, PD


def load_names(path):
    if not os.path.exists(path):
        raise KeyError("%s does not exist ... " % path)
    coco = {}
    with open(path, "rt") as file:
        for index, label in enumerate(file):
            if label:
                coco[index] = label.strip()

    return coco, len(coco)

def iou(gts, pd):
    gts = np.asarray(gts)
    pd = np.asarray(pd)
    xmin = np.maximum(gts[..., 0], pd[0])
    ymin = np.maximum(gts[..., 1], pd[1])
    xmax = np.minimum(gts[..., 2], pd[2])
    ymax = np.minimum(gts[..., 3], pd[3])

    overlap = np.maximum(xmax-xmin, 0.)*np.maximum(ymax-ymin, 0.)
    union = (gts[..., 2]-gts[..., 0])*(gts[..., 3]-gts[..., 1])+(pd[2]-pd[0])*(pd[3]-pd[1])-overlap
    iou = overlap / union

    return iou

def eval(gt, pd):
    npos = 0
    for gt_per_sample in gt:
        npos += len(gt_per_sample)

    confi = [_pd["confi"] for _pd in pd]
    arg_idx = np.argsort(confi)

    tp = [0]*len(pd)
    fp = [0]*len(pd)
    for i, j in enumerate(arg_idx):

        pd_bbox = pd[j]["bbox"]
        sample_id = pd[j]["idx"]

        gt_bboxes = [_gt["bbox"] for _gt in gt[sample_id]]
        score_max = 0.

        if len(gt_bboxes) > 0:
            score = iou(gt_bboxes, pd_bbox)
            gt_idx = np.argmax(score)
            score_max = np.max(score)

        if score_max > MINIOVERLAP:
            if not gt[sample_id][gt_idx]["hit"]:
                gt[sample_id][gt_idx]["hit"] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float32).eps)

    ap = AP(rec, prec)
    return rec, prec, ap


def AP(rec, prec, use_07_metric=False):

    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def report(gts, pds, num_classes):

    rec_list = []
    prec_list = []
    ap_list = []
    for i in range(num_classes):
        # find all same class
        gt_group_by_sample = []
        for gt in gts.values():
            gt_group_by_sample.append(gt[i])

        pd_list = []
        for j, pd in enumerate(pds.values()):
            for bbox in pd[i]:
                bbox["idx"] = j
                pd_list.append(bbox)

        rec, prec, ap = eval(gt_group_by_sample, pd_list)
        rec_list.append(rec)
        prec_list.append(prec)
        ap_list.append(ap)

    return rec_list, prec_list, ap_list

if __name__ == "__main__":

    app.run(main)