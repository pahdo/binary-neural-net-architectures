def binConvolution(input, kernel, phase):
    batchnorm = tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=phase)
    activ = tf.sign(input)
    conv = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding='SAME') # conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')

def binMaxConvolution(input, filter, phase):
    batchnorm = tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=phase)
    activ = tf.sign(input)
    conv = tf.nn.conv2d(input, filter, [1, 1, 1, 1], padding='SAME') # conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    pool = tf.nn.max_pool(input, [3, 3, 3, 3], [2, 2, 2, 2], padding='SAME') # C:add(MaxPooling(3,3,2,2))

"""
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)
"""