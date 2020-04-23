import tensorflow as tf
import os

from tensorflow.keras import models
from absl import app, flags

flags.DEFINE_string('h5_model_path', None, 'Path to H5 Model file')
flags.DEFINE_string('saved_model_path', None, 'Path to Saved Model dir')

FLAGS = flags.FLAGS


def main(_argv):
    model = models.load_model(FLAGS.h5_model_path, compile=False)
    os.makedirs(FLAGS.saved_model_path)
    tf.saved_model.save(model, FLAGS.saved_model_path)


if __name__ == '__main__':
    app.run(main)
