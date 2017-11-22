import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg

slim = tf.contrib.slim

fea_dim = 128

def model(inputs, labels, feature_dim):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = vgg.vgg_16(inputs, spatial_squeeze=False)
        
        global_feature_one = slim.repeat(end_points['vgg_16/pool5'], 2, slim.conv2d, feature_dim, 5, padding='VALID')

        global_feature = slim.conv2d(global_feature_one, num_outputs=feature_dim, kernel_size=3, padding='VALID', scope='global_feature', activation_fn=None)

        local_feature_pool5 = slim.conv2d(end_points['vgg_16/pool5'], num_outputs=feature_dim, kernel_size=[3, 3], padding='SAME', scope='lp5')
        local_feature_pool4 = slim.conv2d(end_points['vgg_16/pool4'], num_outputs=feature_dim, kernel_size=[3, 3], padding='SAME', scope='lp4')
        local_feature_pool3 = slim.conv2d(end_points['vgg_16/pool3'], num_outputs=feature_dim, kernel_size=[3, 3], padding='SAME', scope='lp3')
        local_feature_pool2 = slim.conv2d(end_points['vgg_16/pool2'], num_outputs=feature_dim, kernel_size=[3, 3], padding='SAME', scope='lp2')
        local_feature_pool1 = slim.conv2d(end_points['vgg_16/pool1'], num_outputs=feature_dim, kernel_size=[3, 3], padding='SAME', scope='lp1')

        contract_local_feature_pool5 = contrast_layer(local_feature_pool5)
        contract_local_feature_pool4 = contrast_layer(local_feature_pool4)
        contract_local_feature_pool3 = contrast_layer(local_feature_pool3)
        contract_local_feature_pool2 = contrast_layer(local_feature_pool2)
        contract_local_feature_pool1 = contrast_layer(local_feature_pool1)
        
        local_feature_pool5_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool5, local_feature_pool5], axis=3), feature_dim, kernel_size=5, stride=2)
        local_feature_pool4_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool4, local_feature_pool4], axis=3), feature_dim*2, kernel_size=5, stride=2)
        local_feature_pool3_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool3, local_feature_pool3], axis=3), feature_dim*3, kernel_size=2, stride=2)
        local_feature_pool2_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool2, local_feature_pool2], axis=3), feature_dim*4, kernel_size=5, stride=2)


def contrast_layer(feature_map, kernel_size=3):
    margin = kernel_size/2
    padded_feature_map = tf.pad(feature_map, [[0, 0], [margin, margin], [margin, margin], [0, 0]], 'SYMMETRIC')
    return feature_map - slim.avg_pool2d(padded_feature_map, [kernel_size, kernel_size], stride=1, padding='VALID')

