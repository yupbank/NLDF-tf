import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg

slim = tf.contrib.slim

feature_dim = 128

def nldf(inputs, labels, feature_dim=feature_dim):
    endpoints = {}
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, vg_end_points = vgg.vgg_16(inputs, spatial_squeeze=False)
        vgg_initials = tf.contrib.framework.get_variables_to_restore(include=['vgg_16/conv'])
        endpoints['vggs'] = vgg_initials
        
        global_feature_one = slim.repeat(vg_end_points['vgg_16/pool5'], 
                                        repetitions=2, 
                                        layer=slim.conv2d, 
                                        num_outputs=feature_dim, 
                                        kernel_size=5, 
                                        padding='VALID',
                                        weights_initializer=tf.truncated_normal_initializer(0.01),
                                        )

        global_feature = slim.conv2d(global_feature_one, 
                                     num_outputs=feature_dim, 
                                     kernel_size=3, 
                                     padding='VALID', 
                                     scope='global_feature', 
                                     weights_initializer=tf.truncated_normal_initializer(0.01),
                                     activation_fn=None,
                                     )
        
        global_score = slim.conv2d(global_feature, 
                                   num_outputs=1, 
                                   kernel_size=[1, 1], 
                                   padding='VALID', 
                                   weights_initializer=tf.truncated_normal_initializer(0.01),
                                   activation_fn=None,
                                   )

        local_feature_pool5 = slim.conv2d(
                        vg_end_points['vgg_16/pool5'], 
                        num_outputs=feature_dim, 
                        kernel_size=[3, 3], 
                        padding='SAME', 
                        scope='lp5',
                        weights_initializer=tf.truncated_normal_initializer(0.01),
                        )
        local_feature_pool4 = slim.conv2d(
                        vg_end_points['vgg_16/pool4'], 
                        num_outputs=feature_dim, 
                        kernel_size=[3, 3], 
                        padding='SAME', 
                        scope='lp4',
                        weights_initializer=tf.truncated_normal_initializer(0.01),
                        )
        local_feature_pool3 = slim.conv2d(
                        vg_end_points['vgg_16/pool3'], 
                        num_outputs=feature_dim, 
                        kernel_size=[3, 3], 
                        padding='SAME', 
                        scope='lp3',
                        weights_initializer=tf.truncated_normal_initializer(0.01),
                        )
        local_feature_pool2 = slim.conv2d(
                        vg_end_points['vgg_16/pool2'], 
                        num_outputs=feature_dim, 
                        kernel_size=[3, 3], 
                        padding='SAME', 
                        scope='lp2',
                        weights_initializer=tf.truncated_normal_initializer(0.01),
                        )
        local_feature_pool1 = slim.conv2d(
                        vg_end_points['vgg_16/pool1'], 
                        num_outputs=feature_dim, 
                        kernel_size=[3, 3], 
                        padding='SAME', 
                        scope='lp1',
                        weights_initializer=tf.truncated_normal_initializer(0.01),
                        )

        contract_local_feature_pool5 = contrast_layer(local_feature_pool5)
        contract_local_feature_pool4 = contrast_layer(local_feature_pool4)
        contract_local_feature_pool3 = contrast_layer(local_feature_pool3)
        contract_local_feature_pool2 = contrast_layer(local_feature_pool2)
        contract_local_feature_pool1 = contrast_layer(local_feature_pool1)
        
        local_feature_pool5_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool5, local_feature_pool5], axis=3), feature_dim, kernel_size=5, stride=2)
        local_feature_pool4_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool4, local_feature_pool4, local_feature_pool5_up], axis=3), feature_dim*2, kernel_size=5, stride=2)
        local_feature_pool3_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool3, local_feature_pool3, local_feature_pool4_up], axis=3), feature_dim*3, kernel_size=2, stride=2)
        local_feature_pool2_up = slim.conv2d_transpose(tf.concat([contract_local_feature_pool2, local_feature_pool2, local_feature_pool3_up], axis=3), feature_dim*4, kernel_size=5, stride=2)

        local_feature = slim.conv2d(tf.concat([local_feature_pool1, contract_local_feature_pool1, local_feature_pool2_up], axis=3), num_outputs=feature_dim*5, kernel_size=[1, 1], padding='VALID', scope='lf')
        local_score = slim.conv2d(local_feature, num_outputs=1, kernel_size=[1, 1], padding='VALID', scope='ls')
        score = local_score + global_score
        prob = tf.nn.sigmoid(score)
        endpoints['prob'] = prob 
        cross_entropy_loss = tf.losses.sigmoid_cross_entropy(labels, prob)
        endpoints['cross_entropy_loss'] = cross_entropy_loss
        iou_loss = cal_iou_loss(prob, labels)
        endpoints['iou_loss'] = iou_loss
        loss = cross_entropy_loss+iou_loss
        endpoints['loss'] = loss
        correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        endpoints['accuracy'] = accuracy
        return prob, endpoints


def contrast_layer(feature_map, kernel_size=3):
    margin = kernel_size/2
    padded_feature_map = tf.pad(feature_map, [[0, 0], [margin, margin], [margin, margin], [0, 0]], 'SYMMETRIC')
    return feature_map - slim.avg_pool2d(padded_feature_map, [kernel_size, kernel_size], stride=1, padding='VALID')


def solber_filter(feature_map):
    solber_filter_x = tf.constant([1, 2, 1], dtype=tf.float32)
    solber_filter_y = tf.constant([1, 0, -1], dtype=tf.float32)
    x = tf.reshape(solber_filter_x, (3, 1)) * tf.reshape(solber_filter_y, (1, 3))
    y = tf.reshape(solber_filter_y, (3, 1)) * tf.reshape(solber_filter_x, (1, 3))
    x = tf.reshape(x, [3, 3, 1, 1])
    y = tf.reshape(y, [3, 3, 1, 1])
    padded_feature_map = tf.pad(feature_map, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
    gx = tf.nn.conv2d(padded_feature_map, x, [1, 1, 1, 1], padding='VALID')
    gy = tf.nn.conv2d(padded_feature_map, y, [1, 1, 1, 1], padding='VALID')
    return tf.sqrt(tf.square(gx)+tf.square(gy))


def cal_iou_loss(pred, gt):
    pred_filtered = tf.nn.tanh(solber_filter(pred))
    gt_filtered = tf.nn.tanh(solber_filter(gt))
    inter = tf.reduce_sum(pred_filtered*gt_filtered, 0, keep_dims=True)
    union = tf.reduce_sum(pred_filtered, 0, keep_dims=True)+tf.reduce_sum(gt_filtered, 0, keep_dims=True)
    return tf.reduce_mean(tf.ones_like(inter) - 2 * inter / (union + 1))


def clip_grad(opt, loss, tvars, max_grad_norm):
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    return opt.apply_gradients(zip(grads, tvars))
