import numpy as np
import random

from numpy.random.mtrand import rand

from yolox.tools.voc_to_yolox.utils.voc import parse_voc_annotation

train_ints, train_labels = parse_voc_annotation("/root/deep-learning-ws/data/Annotations/","/dataset/Images/","airbus_train.pkl")
labels = list(train_labels.keys())

lines = []
for img in train_ints:
    line = img['filename'] + " "
    for obj in img['object']:
        line = line + str(obj['xmin']) + "," + str(obj['ymin']) + "," + str(obj['xmax']) + "," + str(obj['ymax']) + "," + str(labels.index(obj['name'])) + " "
    lines.append(line + "\n")

train_valid_split = int(0.8 * len(lines))
random.seed(0)
random.shuffle(lines)
random.seed()

valid_lines = lines[train_valid_split:]
train_lines = lines[:train_valid_split]

with open("dataset_train.txt","w") as file:
    file.writelines(train_lines)

with open("dataset_valid.txt","w") as file:
    file.writelines(valid_lines)