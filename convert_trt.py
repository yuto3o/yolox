# -*- coding: utf-8 -*-
#####################################################################################################
# More Details https://tensorflow.google.cn/api_docs/python/tf/experimental/tensorrt/Converter?hl=en
#####################################################################################################
import os

import cv2
import tensorflow as tf
import numpy as np

from absl import app, flags, logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.experimental import tensorrt as trt

flags.DEFINE_string('weights', None, 'path to weights file')
flags.DEFINE_string('output', None, 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float16', 'quantize mode (int8, float16)')
flags.DEFINE_string('dataset', "./coco_dataset/coco/5k.txt", 'path to dataset')
flags.DEFINE_integer('loop', 10, 'loop')

FLAGS = flags.FLAGS

from tensorflow.python.compiler import tensorrt

def main(_argv):
    params = tf.experimental.tensorrt.ConversionParams(
        precision_mode='FP16',
        # Set this to a large enough number so it can cache all the engines.
        maximum_cached_engines=16)
    converter = trt.Converter(
        input_saved_model_dir="my_dir", conversion_params=params)
    converter.convert()

    # Define a generator function that yields input data, and use it to execute
    # the graph to build TRT engines.
    # With TensorRT 5.1, different engines will be built (and saved later) for
    # different input shapes to the TRTEngineOp.
    def my_input_fn():
        for _ in range(num_runs):
            inp1, inp2 = ...
            yield inp1, inp2

    converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
    converter.save(output_saved_model_dir)  # Generated engines will be saved.


