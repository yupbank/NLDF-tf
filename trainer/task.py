import tensorflow as tf
import model

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


def preprocess_image(file_name):
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


def input_parser(file_name):
    x = tf.string_join([image_location, file_name], '/')
    y = tf.string_join([label_location, file_name], '/')
    x = prepare_image(preprocess_image(x))
    y = prepare_label(preprocess_image(y))
    return x, y

def get_dataset(dataset_location=FLAGS.dataset_location, file_name=FLAGS.train_dataset, batch_size=FLAGS.batch_size):
    dataset = tf.string_join([dataset_location, file_name], '/')
    dataset = tf.contrib.data.TextLineDataset(dataset)
    dataset = dataset.map(input_parser)
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size)
    return dataset



def main(_):
    with tf.Graph().as_default() as g:
        dataset = get_dataset()
        iterator = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        inputs, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(dataset)


        with tf.device(device):
            prob, endpoints = model.nldf(inputs, labels)
            loss = endpoints['loss']
            iou_loss = endpoints['iou_loss']
            cross_entropy_loss = endpoints['cross_entropy_loss']
            accuracy = endpoints['accuracy']
            opt = tf.train.AdamOptimizer(1e-6)
            tvars = tf.trainable_variables()
            train_op = model.clip_grad(opt, loss, tvars, FLAGS.max_grad_norm)

        saver = tf.train.Saver(max_to_keep=4)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            if FLAGS.restore_from_vgg:
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.vgg_model_path, endpoints['vggs'])
                init_fn(sess)

            for i in xrange(FLAGS.num_of_epoch):
                sess.run(train_init_op)
                j = 0
                while True:
                    try:
                        _, dloss, daccuracy, diou, d_cross = sess.run([train_op, loss, accuracy, iou_loss, cross_entropy_loss])
                        print i, j, dloss, daccuracy, diou, d_cross
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break 

                saver.save(sess, FLAGS.checkpoint_dir, global_step=i)


if __name__ == "__main__":
    tf.app.run()
