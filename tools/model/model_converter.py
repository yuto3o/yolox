from absl import app, flags
import os

from tensorflow.keras.models import load_model

from tensorflow import function, TensorSpec
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.io import write_graph

import tf2onnx

flags.DEFINE_string('model_path', '',
                    'full path to the keras model')
flags.DEFINE_bool('output_summary', False,
                  'Enable/Disbale model summary output')
flags.DEFINE_string('output_format', 'weight',
                    'Output format: [weight, frozen_graph, onnx]')
flags.DEFINE_integer(
    'opset', 12, 'If the requested output format is ONNX, the opset number should to be supplied')
FLAGS = flags.FLAGS


def save_frozen_graph(model, model_path, verbose=0):

    # Convert Keras model to ConcreteFunction
    full_model = function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen graph def
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    if verbose > 0:
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

    # Save frozen graph to disk
    write_graph(graph_or_graph_def=frozen_func.graph,
                logdir=os.path.dirname(model_path),
                name=f"{os.path.basename(model_path)}.pb",
                as_text=False)
    # Save its text representation
    write_graph(graph_or_graph_def=frozen_func.graph,
                logdir=os.path.dirname(model_path),
                name=f"{os.path.basename(model_path)}.pbtxt",
                as_text=True)


def main(_argv):
    model = load_model(FLAGS.model_path)

    if FLAGS.output_summary:
        model.summary()

    if FLAGS.output_format == 'weight':
        model.save_weights(f"{FLAGS.model_path}.h5")

    elif FLAGS.output_format == 'frozen_graph':
        save_frozen_graph(model, FLAGS.model_path)

    elif FLAGS.output_format == 'onnx':
        onnx_model_proto, _ = tf2onnx.convert.from_keras(
            model, opset=FLAGS.opset)
        with open(f"{FLAGS.model_path}.onnx", "wb") as f:
            f.write(onnx_model_proto.SerializeToString())

    else:
        print("Requested output format is not supported")


if __name__ == "__main__":
    app.run(main)
