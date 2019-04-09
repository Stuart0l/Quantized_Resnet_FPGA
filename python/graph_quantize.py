from numpy import *
import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python import ops


def get_graph_def_from_file(graph_filepath):
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


transforms = [
    'merge_duplicate_nodes',
    'strip_unused_nodes',
    'quantize_weights',
    'remove_attribute(attribute_name=_class)'
]

quantized_graph_def = TransformGraph(get_graph_def_from_file(os.path.join(os.getcwd(), 'frozen_resnet50.pb')),
                                     ['input'],
                                     ['output'],
                                     transforms)
tf.train.write_graph(quantized_graph_def,
                     os.getcwd(),
                     'quantized_resnet50.pb',
                     False)
