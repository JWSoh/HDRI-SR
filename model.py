from ops import *

class REF_Network(object):
    def __init__(self, x, scale, reuse=False):
        self.input=x
        self.scale=scale
        self.reuse=reuse

        self.build_model(reuse=self.reuse)

    def build_model(self, reuse=False):
        print('Build Model REF')
        with tf.variable_scope('REF', reuse=reuse):
            with tf.variable_scope('First'):
                with tf.variable_scope('Encoder'):
                    with tf.variable_scope('Block1'):
                        self.conv1 = relu(conv2d(self.input, 64, 3, scope='conv1'))

                    with tf.variable_scope('Block2'):
                        self.conv = relu(conv2d(self.conv1, 64, 3, strides=2, scope='conv1'))
                        self.conv2 = relu(conv2d(self.conv, 128, 3, scope='conv2'))

                    with tf.variable_scope('Block3'):
                        self.conv = relu(conv2d(self.conv2, 128, 3, strides=2, scope='conv1'))
                        self.conv3 = relu(conv2d(self.conv, 256, 3, scope='conv2'))

                    with tf.variable_scope('Block4'):
                        self.conv = relu(conv2d(self.conv3, 256, 3, strides=2, scope='conv1'))
                        self.conv4 = relu(conv2d(self.conv, 256, 3, scope='conv2'))

                with tf.variable_scope('Decoder'):
                    with tf.variable_scope('Block5'):
                        self.conv = relu(deconv2d_run(self.conv4, 256, 4, output_shape=self.conv3.get_shape().as_list(), scope='deconv1'))
                        self.conv = tf.concat([self.conv, self.conv3], axis=3)

                        self.conv5 = relu(conv2d(self.conv, 128, 3, scope='conv1'))

                    with tf.variable_scope('Block6'):
                        self.conv = relu(deconv2d_run(self.conv5, 128, 4,output_shape=self.conv2.get_shape().as_list(), scope='deconv1'))
                        self.conv = tf.concat([self.conv, self.conv2], axis=3)

                        self.conv6 = relu(conv2d(self.conv, 64, 3, scope='conv1'))

                    with tf.variable_scope('Block7'):
                        self.conv = relu(deconv2d_run(self.conv6, 64, 4,output_shape=self.conv1.get_shape().as_list(),scope='deconv1'))
                        self.conv = tf.concat([self.conv, self.conv1], axis=3)

                        self.temp = relu(conv2d(self.conv, 64, 3, scope='conv1'))

            with tf.variable_scope('Second'):
                with tf.variable_scope('Encoder'):
                    with tf.variable_scope('Block1'):
                        self.conv1 = relu(conv2d(self.temp, 64, 3, scope='conv1'))

                    with tf.variable_scope('Block2'):
                        self.conv = relu(conv2d(self.conv1, 64, 3, strides=2, scope='conv1'))
                        self.conv2 = relu(conv2d(self.conv, 128, 3, scope='conv2'))

                    with tf.variable_scope('Block3'):
                        self.conv = relu(conv2d(self.conv2, 128, 3, strides=2, scope='conv1'))
                        self.conv3 = relu(conv2d(self.conv, 256, 3, scope='conv2'))

                    with tf.variable_scope('Block4'):
                        self.conv = relu(conv2d(self.conv3, 256, 3, strides=2, scope='conv1'))
                        self.conv4 = relu(conv2d(self.conv, 256, 3, scope='conv2'))

                with tf.variable_scope('Decoder'):
                    with tf.variable_scope('Block5'):
                        self.conv = relu(deconv2d_run(self.conv4, 256, 4, output_shape=self.conv3.get_shape().as_list(), scope='deconv1'))
                        self.conv = tf.concat([self.conv, self.conv3], axis=3)

                        self.conv5 = relu(conv2d(self.conv, 128, 3, scope='conv1'))

                    with tf.variable_scope('Block6'):
                        self.conv = relu(deconv2d_run(self.conv5, 128, 4,output_shape=self.conv2.get_shape().as_list(), scope='deconv1'))
                        self.conv = tf.concat([self.conv, self.conv2], axis=3)

                        self.conv6 = relu(conv2d(self.conv, 64, 3, scope='conv1'))

                    with tf.variable_scope('Block7'):
                        self.conv = relu(deconv2d_run(self.conv6, 64, 4, output_shape=self.conv1.get_shape().as_list(), scope='deconv1'))
                        self.conv = tf.concat([self.conv, self.conv1], axis=3)

                        self.conv = relu(conv2d(self.conv, 64, 3, scope='conv1'))

            self.conv=tf.concat([self.conv, self.temp],axis=-1)

            if self.scale == 4:
                self.conv_up1 = conv2d(self.conv, 64 * self.scale // 2 * self.scale // 2, [3, 3], scope='conv_up1',
                                       activation=None)
                self.conv2_1 = tf.depth_to_space(self.conv_up1, self.scale // 2)

                self.conv_up2 = conv2d(self.conv2_1, 64 * self.scale // 2 * self.scale // 2, [3, 3], scope='conv_up2',
                                       activation=None)
                self.conv2_2 = tf.depth_to_space(self.conv_up2, self.scale // 2)

                self.output = conv2d(self.conv2_2, 1, [3, 3], scope='conv_out', activation=None)

            else:
                self.conv_up1 = conv2d(self.conv, 64 * self.scale * self.scale, [3, 3], scope='conv_up1',
                                       activation=None)
                self.conv2_1 = tf.depth_to_space(self.conv_up1, self.scale)
                self.output = conv2d(self.conv2_1, 1, [3, 3], scope='conv_out', activation=None)

        tf.add_to_collection('InNOut', self.input)
        tf.add_to_collection('InNOut', self.output)