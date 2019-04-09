import tensorflow as tf
import os
import numpy as np

block_sizes = [3, 4, 6, 3]

model = os.path.join(os.getcwd(), "saved_model")
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model)
    graph = tf.get_default_graph()

    # writer = tf.summary.FileWriter('log/', sess.graph)

    all_kernel = []
    dense = []
    bias = []

    variable_names = [v.name for v in tf.global_variables()]
    for item in variable_names:
        if 'resnet_model' in item:
            if 'dense' in item:
                if 'kernel' in item:
                    dense = sess.run(graph.get_tensor_by_name(item))
                else:
                    bias = sess.run(graph.get_tensor_by_name(item))
            else:
                # print(item)
                variable = graph.get_tensor_by_name(item)
                all_kernel.append(sess.run(variable))

    weights = [{'conv': all_kernel[0],
                'bn': [all_kernel[1],
                       all_kernel[2],
                       all_kernel[3],
                       all_kernel[4]]}]
    s = 5
    for i in block_sizes:
        blocks = []
        block = []
        for j in range(1, 4):
            block.append({'conv': all_kernel[s + 5 * j],
                          'bn': [all_kernel[s + 5 * j + 1],
                                 all_kernel[s + 5 * j + 2],
                                 all_kernel[s + 5 * j + 3],
                                 all_kernel[s + 5 * j + 4]]})
        block.append({'conv': all_kernel[s],
                      'bn': [all_kernel[s + 1],
                             all_kernel[s + 2],
                             all_kernel[s + 3],
                             all_kernel[s + 4]]})
        blocks.append(list(block))
        s += 20
        block.clear()
        for j in range(i-1):
            for k in range(3):
                block.append({'conv': all_kernel[s + 15 * j + 5 * k],
                              'bn': [all_kernel[s + 15 * j + 5 * k + 1],
                                     all_kernel[s + 15 * j + 5 * k + 2],
                                     all_kernel[s + 15 * j + 5 * k + 3],
                                     all_kernel[s + 15 * j + 5 * k + 4]]})
            blocks.append(list(block))
            block.clear()
        s += 15 * (i - 1)
        weights.append(list(blocks))
        blocks.clear()

    all_weights = [weights,[dense, bias]]

    np.save('weights.npy', all_weights)



    pass
