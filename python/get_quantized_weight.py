import tensorflow as tf
import os

weight_ops = []
bn_ops = []
min_ops = []
max_ops = []
Min = []
Max = []
dense_op = tf.Operation
dense_max = tf.Operation
dense_min = tf.Operation

with tf.gfile.GFile(os.path.join(os.getcwd(), 'quantized_resnet50.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

with tf.Session(graph=graph) as sess:
    ops = graph.get_operations()
    for op in ops:
        if 'filter' in op.name:
            if 'const' in op.name:
                weight_ops.append(op)
            elif 'min' in op.name:
                min_ops.append(op)
            elif 'max' in op.name:
                max_ops.append(op)
            else:
                pass
        elif 'MatMul' in op.name:
            if 'const' in op.name:
                dense_op = op
            elif 'b_min' in op.name:
                dense_min = op
            elif 'b_max' in op.name:
                dense_max = op
        elif 'scale' in op.name or 'offset' in op.name or 'mean' in op.name or 'variance' in op.name:
            bn_ops.append(op)
        else: pass

# get conv weights
    for w_op in weight_ops:
        w_tensor = graph.get_tensor_by_name(w_op.name + ':0')
        weight = sess.run(w_tensor)
        with tf.gfile.GFile("../weights/conv/pb/" + w_op.name.lstrip('import/') + '.pb', mode='wb') as f:
            f.write(tf.make_tensor_proto(weight, dtype='quint8').SerializeToString())
        '''
        file_name = '/home/stuart/桌面/resnet/weights/conv/' + w_op.name.lstrip('import/') + '.txt'
        file = open(file_name, 'w')
        dim = weight.shape
        for i in range(dim[2]):
            for o in range(dim[3]):
                for h in range(dim[0]):
                    for w in range(dim[1]):
                        file.write(str(weight[h][w][i][o]))
                        if h is dim[0]-1 and w is dim[1]-1:
                            file.write('\n')
                        else:
                            file.write(' ')
        file.close()
        print(w_op.name)
        '''

# get dense weights
    d_tensor = graph.get_tensor_by_name(dense_op.name + ':0')
    d_weight = sess.run(d_tensor)#.tolist()
    with tf.gfile.GFile("../weights/Matmul.pb", mode='wb') as f:
        f.write(tf.make_tensor_proto(d_weight, dtype='quint8').SerializeToString())
    '''
    file_name = '/home/stuart/桌面/resnet/weights/MatMul.txt'
    file = open(file_name, 'w')
    for o in d_weight:
        file.write(str(o).replace(',', '').strip('[]'))
        file.write('\n')
    file.close()
    '''
    bias_tensor = graph.get_tensor_by_name('import/output/y:0')
    bias = sess.run(bias_tensor)
    with tf.gfile.GFile("../weights/bias.pb", mode='wb') as f:
        f.write(tf.make_tensor_proto(bias, dtype='float32').SerializeToString())

# get batchnorm params
    for bn_op in bn_ops:
        bn_tensor = graph.get_tensor_by_name(bn_op.name + ':0')
        bn_param = sess.run(bn_tensor)#.tolist()
        with tf.gfile.GFile("../weights/batchnorm/pb/" + bn_op.name.replace('import/', '') + '.pb', mode='wb') as f:
            f.write(tf.make_tensor_proto(bn_param, dtype='float').SerializeToString())
        '''
        file_name = '/home/stuart/桌面/resnet/weights/batchnorm/' + bn_op.name.replace('import/', '') + '.txt'
        file = open(file_name, 'w')
        file.write(str(bn_param).strip('[]'))
        file.close()
        '''


# get min, max
    max_ops.sort(key=lambda x: int('0'+x.name[14:-4]))
    max_ops.append(dense_max)
    min_ops.sort(key=lambda x: int('0'+x.name[14:-4]))
    min_ops.append(dense_min)
    for i in range(len(max_ops)):
        Min.append(sess.run(graph.get_tensor_by_name(min_ops[i].name + ':0')))
        Max.append(sess.run(graph.get_tensor_by_name(max_ops[i].name + ':0')))
    file = open('/home/stuart/桌面/resnet/weights/conv/max.txt', 'w')
    file.write(str(Max).strip('[]'))
    file.close()
    file = open('/home/stuart/桌面/resnet/weights/conv/min.txt', 'w')
    file.write(str(Min).strip('[]'))
    file.close()
