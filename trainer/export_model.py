import tensorflow as tf
import model
import logging
from functools import partial
import os

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/nldf-network',
                                   """Directory where to training checkpoints saved.""")

tf.app.flags.DEFINE_string('output_dir', '/tmp/nldf-network/exported',
                                   """Directory where to export checkpoints.""")

tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")

tf.app.flags.DEFINE_float('threshold', 0.3,
                            """threshold for bbox generating.""")

tf.app.flags.DEFINE_bool('save_bbox', False,
                            """save feature map or bounding box""")

FLAGS = tf.app.flags.FLAGS
image_size = 352

prepare_image = partial(model.prepare_image, image_size)

def prob_to_bbox(prob, threshold=FLAGS.threshold):
    """cut one bounding box from prob map

    """
    x, y = tf.meshgrid(tf.range(0, prob.shape[1]), tf.range(0, prob.shape[0]))
    background_for_max =tf.zeros_like(prob, dtype=tf.int32) - 1
    background_for_min =tf.zeros_like(prob, dtype=tf.int32) + 1
    box = tf.string_join( 
                    [
                        tf.as_string(tf.reduce_min(tf.where(prob>threshold, x, background_for_min))),
                        tf.as_string(tf.reduce_max(tf.where(prob>threshold, x, background_for_max))),
                        tf.as_string(tf.reduce_min(tf.where(prob>threshold, y, background_for_min))),
                        tf.as_string(tf.reduce_max(tf.where(prob>threshold, y, background_for_max))),
                    ], ',')
    return box


def load_image(file_location):
    image = model.load_image(file_location)
    return prepare_image(image)

def load_shape(file_location):
    image = model.load_image(file_location)
    return tf.shape(image)[:-1]

def reshape_and_save_bbox((prob, shape, file_name)):
    feature_map = tf.image.resize_images(prob, shape)
    bbox = prob_to_bbox(feature_map)
    with tf.control_dependencies([tf.write_file(file_name, bbox)]):
        return tf.constant("success")

def reshape_and_save_featuremap((prob, shape, file_name)):
    feature_map = tf.image.resize_images(prob, shape)
    feature_map = tf.image.convert_image_dtype(feature_map, dtype=tf.uint8)
    content = tf.image.encode_jpeg(feature_map)
    with tf.control_dependencies([tf.write_file(file_name, content)]):
        return tf.constant("success")


def main(_):
    file_locations = tf.placeholder(tf.string, [None])
    output_locations = tf.placeholder(tf.string, [None])
    
    images = tf.map_fn(load_image, file_locations, dtype=tf.float32)
    shapes = tf.map_fn(load_shape, file_locations, dtype=tf.int32)

    resized_images = tf.map_fn(prepare_image, images)

    score = model.nldf(resized_images)

    prob = tf.nn.sigmoid(score)
    
    if FLAGS.save_bbox:
        reshape_and_save  = reshape_and_save_bbox
    else:
        reshape_and_save  = reshape_and_save_featuremap
        
    status = tf.map_fn(reshape_and_save, (prob, shapes, output_locations), dtype=tf.string)

    model_variables = slim.get_trainable_variables()
    saver = tf.train.Saver(model_variables, reshape=False)

    with tf.Session() as sess:
        # Restore variables from training checkpoints.
        saver.restore(sess, FLAGS.checkpoint_dir)

        # Export inference model.
        output_path = os.path.join(
            tf.compat.as_bytes(FLAGS.output_dir),
            tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to %s' % output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)

        # Build the signature_def_map.
        input_file_info = tf.saved_model.utils.build_tensor_info(
            file_locations)
        output_file_info = tf.saved_model.utils.build_tensor_info(
            output_locations)
        output_status_info = tf.saved_model.utils.build_tensor_info(
            status)
        feature_map_tensor_info = tf.saved_model.utils.build_tensor_info(
            prob)

        signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    'image_locations':
                        input_file_info,
                    'output_locations':
                        output_file_info
                },
                outputs={
                    'status':
                        output_status_info,
                    'feature_map':
                        feature_map_tensor_info,
                },
                method_name=tf.saved_model.signature_constants.
                CLASSIFY_METHOD_NAME))

        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print('Successfully exported model to %s' % FLAGS.output_dir)

if __name__ == "__main__":
    tf.app.run()
