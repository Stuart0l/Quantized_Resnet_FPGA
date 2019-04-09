from resnet50 import resnet_50
import tensorflow as tf
import numpy as np
from imagenet_preprocessing import preprocess_image

image_path = '/home/stuart/桌面/ILSVRC2012_img_val/'
image_name = 'ILSVRC2012_val_00000001.JPEG'
filename = image_path + image_name

weights = np.load('weights.npy')
inputs = preprocess_image(filename)

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
results = resnet_50(x, weights)
results = tf.nn.softmax(results)

saver = tf.train.Saver()

with tf.Session() as sess:

    init2 = tf.global_variables_initializer()
    sess.run(init2)

    graph = sess.graph
    graph_def = graph.as_graph_def()

    input_image = sess.run(inputs)
    prob = sess.run(results, feed_dict={x: input_image})
    for k, j in enumerate(prob):
        a = np.argsort(prob[k])
        for i in range(5):
            print([a[-1 - i] - 1, prob[k][a[-1 - i]]])

    output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, graph_def, ['output'])
    f = tf.gfile.GFile('frozen_resnet50.pb', mode='wb')
    f.write(output_graph_def.SerializeToString())
