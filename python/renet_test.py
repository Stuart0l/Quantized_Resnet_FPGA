import tensorflow as tf
import numpy as np
import time
import os
from imagenet_preprocessing import preprocess_image

batch_size = 100
val_size = 50000

image_path = '/home/luoxh/Desktop/ILSVRC2012_img_val/'

file = open('val.txt', 'r').readlines()
filename = []
label = []
for l in file:
    line = l.strip('\n').split(' ')
    filename.append(line[0])
    label.append(int(line[1]))

with tf.gfile.GFile(os.path.join(os.getcwd(), 'frozen_resnet50.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

with tf.Session(graph=graph) as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    output = graph.get_tensor_by_name('import/output:0')
    input = graph.get_tensor_by_name('import/input:0')
    image_batch = []
    top1_correct = 0
    top5_correct = 0
    counter = 0

    for r in range(int(val_size / batch_size)):
        for i in range(batch_size):
            image_batch.append(sess.run(preprocess_image(image_path + filename[batch_size * r + i])))

        start = time.perf_counter()
        prob = sess.run(output, feed_dict={input : image_batch})
        end = time.perf_counter()

        for k in range(batch_size):
            a = np.argsort(prob[k])
            correct = label[batch_size * r + k]
            if (a[-1] - 1) == correct:
                top1_correct += 1
                top5_correct += 1
            else:
                for i in range(1, 5):
                    if (a[-1 - i] - 1) == correct:
                        top5_correct += 1
                        break
            del a

        image_batch[:] = []

        del prob

        counter += batch_size
        print("%d, %d, %.4f, %.4f, %.6f" % (top1_correct, top5_correct, top1_correct/counter, top5_correct/counter, end - start))
