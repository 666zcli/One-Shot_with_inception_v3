#coding:utf-8
import tensorflow as tf
slim=tf.contrib.slim


###################################################################################
#############################    inception_v3 #######################################

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
    """
    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.
    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.
    Here is a mapping from the old_names to the new names:
    Old name                    | New name
    =======================================
    conv0               | Conv2d_1a_3x3
    conv1               | Conv2d_2a_3x3
    conv2               | Conv2d_2b_3x3
    pool1               | MaxPool_3a_3x3
    conv3               | Conv2d_3b_1x1
    conv4               | Conv2d_4a_3x3
    pool2               | MaxPool_5a_3x3
    mixed_35x35x256a    | Mixed_5b
    mixed_35x35x288a    | Mixed_5c
    mixed_35x35x288b    | Mixed_5d
    mixed_17x17x768a    | Mixed_6a
    mixed_17x17x768b    | Mixed_6b
    mixed_17x17x768c    | Mixed_6c
    mixed_17x17x768d    | Mixed_6d
    mixed_17x17x768e    | Mixed_6e
    mixed_8x8x1280a     | Mixed_7a
    mixed_8x8x2048a     | Mixed_7b
    mixed_8x8x2048b     | Mixed_7c
    Args:
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                                stride=1, padding='VALID'):
            # 299 x 299 x 3
            end_point = 'Conv2d_1a_3x3'                 
            # 下面这个卷积会产生: InceptionV3/Conv2d_1a_3x3/weights                   # can be trained
            # InceptionV3/Conv2d_1a_3x3/BatchNorm/beta                             # can be trained
            # InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_mean                # not trainable_variables
            # InceptionV3/Conv2d_1a_3x3/BatchNorm/moving_variance        # not trainable_variables
            net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point, normalizer_fn=slim.batch_norm)     # 299/2=149    
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            # 下面这个卷积会产生: InceptionV3/Conv2d_2a_3x3/weights                # can be trained
            # InceptionV3/Conv2d_2a_3x3/BatchNorm/beta                             # can be trained
            # InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_mean                # not trainable_variables
            # InceptionV3/Conv2d_2a_3x3/BatchNorm/moving_variance        # not trainable_variables
            net = slim.conv2d(net, depth(32), [3, 3], scope=end_point,normalizer_fn=slim.batch_norm)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point, normalizer_fn=slim.batch_norm)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            net = slim.conv2d(net, depth(80), [1, 1], scope=end_point, normalizer_fn=slim.batch_norm)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            net = slim.conv2d(net, depth(192), [3, 3], scope=end_point, normalizer_fn=slim.batch_norm)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 71 x 71 x 192.
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # 35 x 35 x 192.
        # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                                stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1', normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                                                 scope='Conv2d_0b_5x5',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                                 scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                                 scope='Conv2d_0c_3x3',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                                                 scope='Conv_1_0c_5x5',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                                                 scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                                 scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                                 scope='Conv2d_0c_3x3',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                                                 scope='Conv2d_0b_5x5',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                                 scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                                                 scope='Conv2d_0c_3x3',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                                                 padding='VALID', scope='Conv2d_1a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                                                 scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                                                 padding='VALID', scope='Conv2d_1a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                                         scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                                                 scope='Conv2d_0b_1x7',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                                 scope='Conv2d_0c_7x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                                                 scope='Conv2d_0b_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                                                 scope='Conv2d_0c_1x7',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                                                 scope='Conv2d_0d_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                                 scope='Conv2d_0e_1x7',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                                                 scope='Conv2d_0b_1x7',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                                 scope='Conv2d_0c_7x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                                 scope='Conv2d_0b_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                                                 scope='Conv2d_0c_1x7',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                                 scope='Conv2d_0d_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                                 scope='Conv2d_0e_1x7',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                                                 scope='Conv2d_0b_1x7',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                                 scope='Conv2d_0c_7x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                                 scope='Conv2d_0b_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                                                 scope='Conv2d_0c_1x7',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                                                 scope='Conv2d_0d_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                                 scope='Conv2d_0e_1x7',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                                                 scope='Conv2d_0b_1x7',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                                 scope='Conv2d_0c_7x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                                                 scope='Conv2d_0b_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                                 scope='Conv2d_0c_1x7',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                                                 scope='Conv2d_0d_7x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                                                 scope='Conv2d_0e_1x7',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                                                 scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                                                 padding='VALID', scope='Conv2d_1a_3x3',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                                                 scope='Conv2d_0b_1x7',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                                                 scope='Conv2d_0c_7x1',normalizer_fn=slim.batch_norm)
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                                                 padding='VALID', scope='Conv2d_1a_3x3',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                                         scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3',normalizer_fn=slim.batch_norm),
                            slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1',normalizer_fn=slim.batch_norm)])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(
                            branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
                    branch_2 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3',normalizer_fn=slim.batch_norm),
                            slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1',normalizer_fn=slim.batch_norm)])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                            branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_1 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3',normalizer_fn=slim.batch_norm),
                            slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1',normalizer_fn=slim.batch_norm)])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1',normalizer_fn=slim.batch_norm)
                    branch_2 = slim.conv2d(
                            branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3',normalizer_fn=slim.batch_norm)
                    branch_2 = tf.concat(axis=3, values=[
                            slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3',normalizer_fn=slim.batch_norm),
                            slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1',normalizer_fn=slim.batch_norm)])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                            branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1',normalizer_fn=slim.batch_norm)
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint: return net, end_points
        raise ValueError('Unknown final endpoint %s' % final_endpoint)




def inception_v3(inputs,
                 final_endpoint='Mixed_7c',
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='InceptionV3'):
    """Inception model from http://arxiv.org/abs/1512.00567.
    """
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes],
                                                 reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                                                is_training=is_training):
            net, end_points = inception_v3_base(inputs, 
                                                final_endpoint=final_endpoint,
                                                scope=scope, 
                                                min_depth=min_depth,
                                                depth_multiplier=depth_multiplier)
            # Auxiliary Head logits
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],stride=1, padding='SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',scope='AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],weights_initializer=trunc_normal(0.01),
                                            activation_fn=None,#tf.nn.relu,
                                            scope='Conv2d_1b_1x1',normalizer_fn=slim.batch_norm)
                    # Shape of feature map before the final layer.
                    kernel_size = _reduced_kernel_size_for_small_input(
                            aux_logits, [5, 5])
                    aux_logits = slim.conv2d(aux_logits, depth(768), kernel_size,weights_initializer=trunc_normal(0.01),
                                            padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size),normalizer_fn=slim.batch_norm)
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None,#tf.nn.relu,
                                            weights_initializer=trunc_normal(0.01),scope='Conv2d_2b_1x1',normalizer_fn=slim.batch_norm)
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits
            # Final pooling and prediction
            with tf.variable_scope('Logits'):
                kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
                net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                                            scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                # 1 x 1 x 2048
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,#tf.nn.relu,
                                                         weights_initializer=trunc_normal(0.01),
                                                         scope='Conv2d_1c_1x1',normalizer_fn=slim.batch_norm)
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                # 1000
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points




def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                                             min(shape[2], kernel_size[1])]
    return kernel_size_out


inception_v3.default_image_size = 299

#############################    inception_v3 #######################################
###################################################################################

