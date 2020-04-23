import tensorflow as tf

from tensorflow.keras import models
from absl import app, flags

flags.DEFINE_string('saved_model_path', None, 'Path to Saved Model dir')
flags.DEFINE_string('tflite_path', None, 'Path to TFLite file path')

FLAGS = flags.FLAGS


def main(_argv):
    model = models.load_model(FLAGS.saved_model_path)

    wrapped_model = tf.function(lambda input_data: model(input_data))
    input_spec = tf.TensorSpec((None, 416, 416, 3), model.inputs[0].dtype)
    concrete_func = wrapped_model.get_concrete_function(input_spec)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    open(FLAGS.tflite_path, "wb").write(tflite_quant_model)


if __name__ == '__main__':
    app.run(main)
