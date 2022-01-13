from absl import app, flags
import random

from utils.voc import parse_voc_annotation

flags.DEFINE_string('annotations', '/dataset/Annotations/',
                    'path to annotation folder')
flags.DEFINE_string('images', '/dataset/Images/',
                    'path to image folder')
flags.DEFINE_string('cache', 'data.pkl',
                    'path to the cache file')
flags.DEFINE_float('split_ratio', 0.8,
                   'Split ratio between train and validation data')
flags.DEFINE_string('train_output', 'dataset_train.txt',
                    'Train data')
flags.DEFINE_string('valid_output', 'dataset_valid.txt',
                    'Validation data')
FLAGS = flags.FLAGS


def main(_argv):
    train_ints, train_labels = parse_voc_annotation(
        FLAGS.annotations, FLAGS.images, FLAGS.cache)
    labels = list(train_labels.keys())

    lines = []
    for img in train_ints:
        line = img['filename'] + " "
        for obj in img['object']:
            line = line + str(obj['xmin']) + "," + str(obj['ymin']) + "," + str(
                obj['xmax']) + "," + str(obj['ymax']) + "," + str(labels.index(obj['name'])) + " "
        lines.append(line + "\n")

    train_valid_split = int(FLAGS.split_ratio * len(lines))
    random.seed(0)
    random.shuffle(lines)
    random.seed()

    valid_lines = lines[train_valid_split:]
    train_lines = lines[:train_valid_split]

    with open(FLAGS.train_output, "w") as file:
        file.writelines(train_lines)

    with open(FLAGS.valid_output, "w") as file:
        file.writelines(valid_lines)


if __name__ == "__main__":
    app.run(main)
