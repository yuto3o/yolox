# -*- coding: utf-8 -*-
import yaml
import os
import numpy as np
import pprint

from core.utils import decode_name


def _decode_yaml_tuple(tuple_str):
    return np.array(list(map(lambda x: list(map(int, str.split(x, ','))), tuple_str.split())))


def decode_cfg(path):
    print('Loading config from', path)
    if not os.path.exists(path):
        raise KeyError('%s does not exist ... ' % path)

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f.read())

        # some fields need to be decoded
        cfg['yolo']['strides'] = list(map(int, cfg['yolo']['strides'].split(',')))
        cfg['yolo']['mask'] = _decode_yaml_tuple(cfg['yolo']['mask'])
        cfg['yolo']['anchors'] = _decode_yaml_tuple(cfg['yolo']['anchors'])
        cfg['yolo']['names'] = decode_name(cfg['yolo']['name_path'])
        cfg['yolo']['num_classes'] = len(cfg['yolo']['names'])

        cfg['train']['image_size'] = list(map(int, cfg['train']['image_size'].split(',')))
        cfg['test']['image_size'] = list(map(int, cfg['test']['image_size'].split(',')))


        pprint.pprint(cfg)

    return cfg
