import tensorflow as tf
import numpy as np
import os
from imagenet_preprocessing import preprocess_image

image_path = '/home/stuart/桌面/ILSVRC2012_img_val/'
image_name = 'ILSVRC2012_val_00000004.JPEG'
filename = image_path + image_name

inputs = preprocess_image(filename)

with tf.Session() as sess1:
    input_image = sess1.run(inputs)

with tf.gfile.GFile(os.path.join(os.getcwd(), 'quantized_resnet50.pb'), 'rb') as f:

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
    for op in graph.get_operations():
        print(op.name)

with tf.Session(graph=graph) as sess:
    summarywritter = tf.summary.FileWriter('log/', graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    output = graph.get_tensor_by_name('import/output:0')
    input = graph.get_tensor_by_name('import/input:0')
    output1 = sess.run(output, feed_dict={input: input_image})
    for k, j in enumerate(output1):
        a = np.argsort(output1[k])
        for i in range(5):
            print([a[-1 - i] - 1, output1[k][a[-1 - i]]])
