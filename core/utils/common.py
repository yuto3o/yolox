# -*- coding: utf-8 -*-
def decode_name(name_path):
    with open(name_path, 'r') as f:
        lines = f.readlines()
    name = []
    for line in lines:
        line = line.strip()
        if line:
            name.append(line)
    return name


def decode_annotation(anno_path, type='y_true'):
    with open(anno_path, 'r') as f:
        lines = f.readlines()
    anno = []
    for line in lines:
        line = line.strip()
        if line:
            anno.append(decode_line(line, type))
    return anno


def decode_line(line, type):
    if type == 'y_true':
        return decode_line_y_true(line)
    elif type == 'y_pred':
        return decode_line_y_pred(line)
    else:
        raise NotImplementedError(type)


def decode_line_y_pred(line):
    """
    format
    path x1,y1,x2,y2,label x1,y1,x2,y2,label...
    :param line:
    :return: {'path':str, 'bboxes':list, (x1,y1,x2,y2), 'labels':list }
    """
    items = line.split()
    path = items[0]
    items = items[1:]

    bboxes = []
    labels = []
    confis = []
    for item in items:
        if not item:
            continue
        x1, y1, x2, y2, label, confi = item.split(',')
        x1, y1, x2, y2, label, confi = float(x1), float(y1), float(x2), float(y2), int(label), float(confi)
        bboxes.append([x1, y1, x2, y2])
        labels.append(label)
        confis.append(confi)

    return path, bboxes, labels, confis


def decode_line_y_true(line):
    """
    format
    path x1,y1,x2,y2,label x1,y1,x2,y2,label...
    :param line:
    :return: {'path':str, 'bboxes':list, (x1,y1,x2,y2), 'labels':list }
    """
    items = line.split()
    path = items[0]
    items = items[1:]

    bboxes = []
    labels = []
    for item in items:
        if not item:
            continue
        x1, y1, x2, y2, label = item.split(',')
        x1, y1, x2, y2, label = float(x1), float(y1), float(x2), float(y2), float(label)
        bboxes.append([x1, y1, x2, y2])
        labels.append(label)

    return path, bboxes, labels
