import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg

slim = tf.contrib.slim

fea_dim = 128

def model(inputs, labels):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        outputs, end_points = vgg.vgg_16(inputs)
        
        global_feature_one = slim.repeat(end_points['vgg_16/pool5'], 2, slim.conv2d, fea_dim, [5, 5], padding='VALID', 
                                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

        global_feature = slim.conv2d(global_feature_one,
                                         num_outputs=fea_dim,
                                         keneral_size=[5, 5], 
                                         padding='VALID',
                                         scope='global_feature',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         activation_fn=None
                                         )

        local_feature_pool5 = slim.conv2d(end_points['vgg_16/pool5'], 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='lp5',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool4 = slim.conv2d(end_points['vgg_16/pool4'], 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='lp4',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool3 = slim.conv2d(end_points['vgg_16/pool3'], 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='lp3',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool2 = slim.conv2d(end_points['vgg_16/pool2'], 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='lp2',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )
        local_feature_pool1 = slim.conv2d(end_points['vgg_16/pool1'], 
                                         num_outputs=fea_dim,
                                         keneral_size=[3, 3], 
                                         padding='SAME',
                                         scope='lp1',
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                         )



def contrast_layer(feature_map, keneral_size):
    return feature_map - slim.avg_pool_2d(feature_map, [3, 3])


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
