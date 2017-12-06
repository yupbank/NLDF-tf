import tensorflow as tf
import model
import logging
from functools import partial

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
writer_location = FLAGS.checkpoint_dir+'/eval'


prepare_image = partial(model.prepare_image, image_size)
prepare_label = partial(model.prepare_label, label_size)

def load_image_and_label(file_name):
    x = tf.string_join([image_location, file_name], '/')
    y = tf.string_join([label_location, file_name], '/')
    return model.load_image(x), model.load_image(y)

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

        with tf.device(device):
            scores = model.nldf(inputs)

        prob, end_points = model.loss(scores, labels)

        vgg_variables = tf.contrib.framework.get_trainable_variables('vgg_16/conv')

        total_loss = end_points['loss']
        accuracy = end_points['accuracy']
        optimizer = tf.train.AdamOptimizer(1e-6)
        train_op = slim.learning.create_train_op(
                          total_loss,
                          optimizer,
                          clip_gradient_norm=FLAGS.max_grad_norm)

        saver = tf.train.Saver(max_to_keep=4)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                            tf.nn.zero_fraction(x)))
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summaries.add(tf.summary.scalar('total_loss', total_loss))
        summaries.add(tf.summary.image('Truth', labels*255.0))
        summaries.add(tf.summary.image('Predicted', prob*255.0))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(writer_location, sess.graph)
            if FLAGS.restore_from_vgg:
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.vgg_model_path, vgg_variables)
                init_fn(sess)

            j = 0
            for i in xrange(FLAGS.num_of_epoch):
                sess.run(train_init_op)
                while True:
                    try:
                        if j % 10 == 0:
                            _, summary_str = sess.run([train_op, summary_op])
                            summary_writer.add_summary(summary_str, j)
                            summary_writer.flush()
                        else:
                            _, dloss, daccuracy = sess.run([train_op, total_loss, accuracy])
                            logging.info('Epoch: %s, Batch: %s, Loss: %s, Accuracy: %s'%(i, j, dloss, daccuracy))
                            #print('Epoch: %s, Batch: %s, Loss: %s, Accuracy: %s'%(i, j, dloss, daccuracy))
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break

                saver.save(sess, FLAGS.checkpoint_dir, global_step=i)


if __name__ == "__main__":
    tf.app.run()
