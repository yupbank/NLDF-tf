import tensorflow as tf


slim = tf.contrib.slim

fea_dim = 128

def model(inputs, labels):
    with slim.arg_scope(slim.net.vgg.vgg_arg_scope()):
        outputs, end_points = slim.net.vgg.vgg_16(inputs)
    
        global_feature_one = slim.conv2d(vgg.pool5, 
                                         num_outputs=fea_dim,
                                         keneral_size=[5, 5], 
                                         padding='VALID',
                                         scope='global_feature_one',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )

        global_feature_two = slim.conv2d(global_feature_one,
                                         num_outputs=fea_dim,
                                         keneral_size=[5, 5], 
                                         padding='VALID',
                                         scope='global_feature_two',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        global_feature = slim.conv2d(global_feature_two,
                                         num_outputs=fea_dim,
                                         keneral_size=[5, 5], 
                                         padding='VALID',
                                         scope='global_feature',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         activation_fn=None
                                         )

        local_feature_pool5 = slim.conv2d(vgg.pool5, 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='global_feature_two',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool4 = slim.conv2d(vgg.pool4, 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='global_feature_two',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool3 = slim.conv2d(vgg.pool3, 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='global_feature_two',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool2 = slim.conv2d(vgg.pool2, 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='global_feature_two',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )


                                            [5, 5, 512, fea_dim], 0.01,
                                                    padding='VALID', name='Fea_Global_1'))
        self.Fea_Global_2 = tf.nn.relu(self.Conv_2d(self.Fea_Global_1, [5, 5, fea_dim, fea_dim], 0.01,
                                                    padding='VALID', name='Fea_Global_2'))
        self.Fea_Global = self.Conv_2d(self.Fea_Global_2, [3, 3, fea_dim, fea_dim], 0.01,
                                       padding='VALID', name='Fea_Global')

        #Local Score
        self.Fea_P5 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3, 3, 512, fea_dim], 0.01, padding='SAME', name='Fea_P5'))
        self.Fea_P4 = tf.nn.relu(self.Conv_2d(vgg.pool4, [3, 3, 512, fea_dim], 0.01, padding='SAME', name='Fea_P4'))
        self.Fea_P3 = tf.nn.relu(self.Conv_2d(vgg.pool3, [3, 3, 256, fea_dim], 0.01, padding='SAME', name='Fea_P3'))
        self.Fea_P2 = tf.nn.relu(self.Conv_2d(vgg.pool2, [3, 3, 128, fea_dim], 0.01, padding='SAME', name='Fea_P2'))
        self.Fea_P1 = tf.nn.relu(self.Conv_2d(vgg.pool1, [3, 3, 64, fea_dim], 0.01, padding='SAME', name='Fea_P1'))

        self.Fea_P5_LC = self.Contrast_Layer(self.Fea_P5, 3)
        self.Fea_P4_LC = self.Contrast_Layer(self.Fea_P4, 3)
        self.Fea_P3_LC = self.Contrast_Layer(self.Fea_P3, 3)
        self.Fea_P2_LC = self.Contrast_Layer(self.Fea_P2, 3)
        self.Fea_P1_LC = self.Contrast_Layer(self.Fea_P1, 3)

        #Deconv Layer
        self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P5, self.Fea_P5_LC], axis=3),
                                                   [1, 22, 22, fea_dim], 5, 2, name='Fea_P5_Deconv'))
        self.Fea_P4_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P4, self.Fea_P4_LC, self.Fea_P5_Up], axis=3),
                                                   [1, 44, 44, fea_dim*2], 5, 2, name='Fea_P4_Deconv'))
        self.Fea_P3_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P3, self.Fea_P3_LC, self.Fea_P4_Up], axis=3),
                                                   [1, 88, 88, fea_dim*3], 5, 2, name='Fea_P3_Deconv'))
        self.Fea_P2_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P2, self.Fea_P2_LC, self.Fea_P3_Up], axis=3),
                                                   [1, 176, 176, fea_dim*4], 5, 2, name='Fea_P2_Deconv'))

        self.Local_Fea = self.Conv_2d(tf.concat([self.Fea_P1, self.Fea_P1_LC, self.Fea_P2_Up], axis=3),
                                      [1, 1, fea_dim*6, fea_dim*5], 0.01, padding='VALID', name='Local_Fea')
        self.Local_Score = self.Conv_2d(self.Local_Fea, [1, 1, fea_dim*5, 2], 0.01, padding='VALID', name='Local_Score')

        self.Global_Score = self.Conv_2d(self.Fea_Global,
                                         [1, 1, fea_dim, 2], 0.01, padding='VALID', name='Global_Score')

        self.Score = self.Local_Score + self.Global_Score
        self.Score = tf.reshape(self.Score, [-1,2])

        self.Prob = tf.nn.softmax(self.Score)
