# -*- coding: utf-8 -*-
import yaml
import os

from .utils import decode_yaml_tuple, load_names, load_annotations


def load(path):
    print("Loading config from", path)
    if not os.path.exists(path):
        raise KeyError("%s does not exist ... " % path)

    with open(path, "r") as f:
        cfg = yaml.safe_load(f.read())

        # some fields need to be decoded
        cfg["yolo"]["classes"] = load_names(cfg["yolo"]["classes"])
        cfg["yolo"]["num_classes"] = len(cfg["yolo"]["classes"])

        cfg["yolo"]["strides"] = list(map(int, cfg["yolo"]["strides"].split(",")))
        cfg["yolo"]["mask"] = decode_yaml_tuple(cfg["yolo"]["mask"])
        cfg["yolo"]["anchors"] = decode_yaml_tuple(cfg["yolo"]["anchors"])

        cfg["train"]["annotations"] = load_annotations(cfg["train"]["annot_path"])
        cfg["train"]["image_size"] = list(map(int, cfg["train"]["image_size"].split(",")))
        cfg["train"]["lr_init"] = float(cfg["train"]["lr_init"])
        cfg["train"]["lr_end"] = float(cfg["train"]["lr_end"])

        cfg["test"]["image_size"] = list(map(int, cfg["test"]["image_size"].split(",")))

    return cfg
