import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets import vgg

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

slim = tf.contrib.slim

feature_dim = 128


def load_image(file_name):
    image = tf.image.decode_jpeg(tf.read_file(file_name), dct_method="INTEGER_ACCURATE")
    return tf.image.convert_image_dtype(image, tf.float32)


def prepare_image(image_size, image):
    image = tf.image.resize_images(image, (image_size, image_size))
    return image - [_R_MEAN, _G_MEAN, _B_MEAN]


def prepare_label(label_size, image):
    label = tf.image.resize_images(image, (label_size, label_size))
    return label/255.0


def load_vgg(inputs):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, vg_end_points = vgg.vgg_16(inputs, spatial_squeeze=False)
        pool5, pool4, pool3, pool2, pool1 = vg_end_points['vgg_16/pool5'], vg_end_points['vgg_16/pool4'], vg_end_points['vgg_16/pool3'], vg_end_points['vgg_16/pool2'], vg_end_points['vgg_16/pool1']
    return pool5, pool4, pool3, pool2, pool1


def contrast_layer(feature_map, kernel_size=3):
    margin = kernel_size/2
    padded_feature_map = tf.pad(feature_map, [[0, 0], [margin, margin], [margin, margin], [0, 0]], 'SYMMETRIC')
    return feature_map - slim.avg_pool2d(padded_feature_map, [kernel_size, kernel_size], stride=1, padding='VALID')


def sobel_filter(feature_map):
    sobel_filter_x = tf.constant([1, 2, 1], dtype=tf.float32)
    sobel_filter_y = tf.constant([1, 0, -1], dtype=tf.float32)
    x = tf.reshape(sobel_filter_x, (3, 1)) * tf.reshape(sobel_filter_y, (1, 3))
    y = tf.reshape(sobel_filter_y, (3, 1)) * tf.reshape(sobel_filter_x, (1, 3))
    x = tf.reshape(x, [3, 3, 1, 1])
    y = tf.reshape(y, [3, 3, 1, 1])
    padded_feature_map = tf.pad(feature_map, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
    gx = tf.nn.conv2d(padded_feature_map, x, [1, 1, 1, 1], padding='VALID')
    gy = tf.nn.conv2d(padded_feature_map, y, [1, 1, 1, 1], padding='VALID')
    return tf.sqrt(tf.square(gx)+tf.square(gy))


def cal_iou_loss(pred, gt, contour_th=1.5):
    squash_channels = lambda r: tf.reduce_sum(sobel_filter(r), 3, keep_dims=True)
    pred = squash_channels(pred)
    gt = squash_channels(gt)

    pred_filtered = tf.nn.tanh(pred)
    gt_filtered = tf.where(gt>contour_th, tf.ones_like(gt), tf.zeros_like(gt))

    inter = tf.reduce_sum(pred_filtered*gt_filtered, [1, 2, 3])
    union = tf.reduce_sum(tf.square(pred_filtered), [1, 2, 3]) + tf.reduce_sum(tf.square(gt_filtered), [1, 2, 3])

    return tf.reduce_mean(tf.ones_like(inter) - 2 * inter / (union + 1), name='iou_loss')


def local_global_score(pool5, pool4, pool3, pool2, pool1, feature_dim=feature_dim):
    with slim.arg_scope([slim.conv2d], padding='VALID',
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       biases_initializer=tf.constant_initializer(0.0)):
        Fea_Global_2 = slim.repeat(pool5, repetitions=2, layer=slim.conv2d, num_outputs=feature_dim, kernel_size=5, scope='Fea_Global')

        with slim.arg_scope([slim.conv2d], activation_fn=None):
            Fea_Global = slim.conv2d(Fea_Global_2, num_outputs=feature_dim, kernel_size=3, scope='Fea_Global')
            Global_Score = slim.conv2d(Fea_Global, num_outputs=1, kernel_size=1, scope='Global_Score',)

        with slim.arg_scope([slim.conv2d], padding='SAME', kernel_size=3, num_outputs=feature_dim):
            Fea_P5 = slim.conv2d(pool5, scope='Fea_P5')
            Fea_P4 = slim.conv2d(pool4, scope='Fea_P4')
            Fea_P3 = slim.conv2d(pool3, scope='Fea_P3')
            Fea_P2 = slim.conv2d(pool2, scope='Fea_P2')
            Fea_P1 = slim.conv2d(pool1, scope='Fea_P1')

    Fea_P5_LC = contrast_layer(Fea_P5)
    Fea_P4_LC = contrast_layer(Fea_P4)
    Fea_P3_LC = contrast_layer(Fea_P3)
    Fea_P2_LC = contrast_layer(Fea_P2)
    Fea_P1_LC = contrast_layer(Fea_P1)

    with slim.arg_scope([slim.conv2d_transpose], kernel_size=5, stride=2):
        Fea_P5_Up = slim.conv2d_transpose(tf.concat([Fea_P5, Fea_P5_LC], axis=3),
                                                       feature_dim,
                                                       scope='Fea_P5_Deconv')
        Fea_P4_Up = slim.conv2d_transpose(tf.concat([Fea_P4, Fea_P4_LC, Fea_P5_Up], axis=3),
                                                        feature_dim*2,
                                                        scope='Fea_P4_Deconv')
        Fea_P3_Up = slim.conv2d_transpose(tf.concat([Fea_P3, Fea_P3_LC, Fea_P4_Up], axis=3),
                                                        feature_dim*3,
                                                        scope='Fea_P3_Deconv')
        Fea_P2_Up = slim.conv2d_transpose(tf.concat([Fea_P2, Fea_P2_LC, Fea_P3_Up], axis=3),
                                                        feature_dim*4,
                                                        scope='Fea_P2_Deconv')

    with slim.arg_scope([slim.conv2d], padding='VALID',
                                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                       biases_initializer=tf.constant_initializer(0.0),
                                       kernel_size=1,
                                       activation_fn=None):
        Local_Fea = slim.conv2d(tf.concat([Fea_P1, Fea_P1_LC, Fea_P2_Up], axis=3),
                                    num_outputs=feature_dim*5, 
                                    scope='Local_Fea')

        Local_Score = slim.conv2d(Local_Fea, num_outputs=1, scope='Local_Score')

    return Local_Score, Global_Score

def nldf(inputs, feature_dim=feature_dim):
    pool5, pool4, pool3, pool2, pool1 = load_vgg(inputs)
    local_score, global_score = local_global_score(pool5, pool4, pool3, pool2, pool1, feature_dim)
    score = local_score + global_score
    return score


def loss(scores, labels, contour_th=1.5):
    endpoints = {}

    prob = tf.nn.sigmoid(scores)
    endpoints['prob'] = prob 

    multi_class_labels = tf.where(labels > 0.0, tf.ones_like(labels), tf.zeros_like(labels))
    cross_entropy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = multi_class_labels, logits=scores)
    endpoints['cross_entropy_loss'] = cross_entropy_loss

    iou_loss = cal_iou_loss(prob, labels, contour_th)
    tf.losses.add_loss(iou_loss)
    endpoints['iou_loss'] = iou_loss

    loss = cross_entropy_loss+iou_loss
    endpoints['loss'] = loss
    
    accuracy = tf.reduce_mean(tf.losses.absolute_difference(labels=labels, predictions=prob))
    endpoints['accuracy'] = accuracy

    return prob, endpoints
