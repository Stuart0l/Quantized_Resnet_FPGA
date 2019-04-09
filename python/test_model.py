import tensorflow as tf
import os
import numpy as np
from imagenet_preprocessing import preprocess_image

model = os.path.join(os.getcwd(), "saved_model")
image_path = '/home/stuart/桌面/ILSVRC2012_img_val/'
image_name = 'ILSVRC2012_val_00000004.JPEG'
filename = image_path + image_name

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model)
    graph = tf.get_default_graph()

    inputs = sess.run(preprocess_image(filename))

    output = tf.squeeze(graph.get_tensor_by_name("softmax_tensor:0"))
    prob = sess.run(output, feed_dict={"map/TensorArrayStack/TensorArrayGatherV3:0": inputs})
    a = np.argsort(prob)
    for i in range(5):
        print([a[-1 - i] - 1, prob[a[-1 - i]]]) # resnet outputs label[1, 1000], caffe use label[0, 999]
