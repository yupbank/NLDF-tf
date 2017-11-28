import tensorflow as tf
import model
import logging


slim = tf.contrib.slim

image_size = 352
label_size = image_size/2
train_dataset = 'train.txt'

tf.app.flags.DEFINE_string('train_dataset', 'train.txt',
                                   """Train dataset""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/nldf-network',
                                   """Directory where to save training checkpoints.""")
tf.app.flags.DEFINE_string('vgg_model_path', '/tmp/vgg-network',
                                   """Directory where to load vgg variables""")
tf.app.flags.DEFINE_bool('restore_from_vgg', True,
                                    """Restore from vgg""")
tf.app.flags.DEFINE_bool('use_augmentation', True,
                                    """Restore from vgg""")
tf.app.flags.DEFINE_string('dataset_location', '/Users/pengyu/Downloads/HKU-IS',
                                   """Directory where to the data""")
tf.app.flags.DEFINE_integer('num_of_epoch', 1,
                                   """num of epoch""")
tf.app.flags.DEFINE_integer('num_of_batch', 1,
                                   """num of batch""")
tf.app.flags.DEFINE_float('max_grad_norm', 1.0,
                                   """ max grad norm""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                                   """batch size""")
tf.app.flags.DEFINE_bool('use_gpu', False,
                                    """Restore from vgg""")


FLAGS = tf.app.flags.FLAGS

device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'

image_location = FLAGS.dataset_location+'/imgs'
label_location = FLAGS.dataset_location+'/gt'


def load_image(file_name):
    image = tf.image.decode_jpeg(tf.read_file(file_name), dct_method="INTEGER_ACCURATE")
    return tf.image.convert_image_dtype(image, tf.float32)


def prepare_image(image):
    image = tf.reverse(image, [-1])
    image = tf.image.resize_images(image, (image_size, image_size))
    image = image - tf.constant([103.939, 116.779, 123.68])
    return image


def prepare_label(image):
    label = tf.image.resize_images(image, (label_size, label_size))
    return label/255.

def load_image_and_label(file_name):
    x = tf.string_join([image_location, file_name], '/')
    y = tf.string_join([label_location, file_name], '/')
    return load_image(x), load_image(y)

def input_parser(file_name):
    x, y = load_image_and_label(file_name)
    x = prepare_image(x)
    y = prepare_label(y)
    return x, y

def augmented_input_parser(file_name):
    res = []
    x_img, y_img = load_image_and_label(file_name)
    x1 = prepare_image(x_img)
    y1 = prepare_label(y_img)
    res.append((x1, y1))

    flipped_x = tf.image.flip_left_right(x_img)
    flipped_y = tf.image.flip_left_right(y_img)

    x2 = prepare_image(flipped_x)
    y2 = prepare_label(flipped_y)
    res.append((x2, y2))
    return res


def get_dataset(dataset_location=FLAGS.dataset_location, file_name=FLAGS.train_dataset, batch_size=FLAGS.batch_size):
    dataset = tf.string_join([dataset_location, file_name], '/')
    dataset = tf.contrib.data.TextLineDataset(dataset)

    dataset = dataset.map(input_parser)

    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    return dataset



def main(_):
    with tf.Graph().as_default() as g:
        dataset = get_dataset()
        iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        inputs, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(dataset)

        with tf.device(tf.train.replica_device_setter()):
            prob, endpoints = model.nldf(inputs, labels)

        vgg_variables = tf.contrib.framework.get_trainable_variables('vgg_16/conv')

        loss = endpoints['loss']
        accuracy = endpoints['accuracy']
        optimizer = tf.train.AdamOptimizer(1e-6)
        train_op = slim.learning.create_train_op(
                          loss,
                          optimizer,
                          clip_gradient_norm=FLAGS.max_grad_norm)

        saver = tf.train.Saver(max_to_keep=4)

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            if FLAGS.restore_from_vgg:
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.vgg_model_path, vgg_variables)
                init_fn(sess)

            for i in xrange(FLAGS.num_of_epoch):
                sess.run(train_init_op)
                j = 0
                while True:
                    try:
                        _, dloss, daccuracy = sess.run([train_op, loss, accuracy])
                        logging.warn('%s, %s, %s, %s'%(i, j, dloss, daccuracy))
                        print i, j, dloss, daccuracy
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break

                saver.save(sess, FLAGS.checkpoint_dir, global_step=i)


if __name__ == "__main__":
    tf.app.run()
