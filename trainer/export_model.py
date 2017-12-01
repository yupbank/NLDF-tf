import tensorflow as tf
import model
import task
import logging




tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/nldf-network',
                                   """Directory where to training checkpoints saved.""")

tf.app.flags.DEFINE_string('export_dir', '/tmp/nldf-network/exported',
                                   """Directory where to export checkpoints.""")

FLAGS = tf.app.flags.FLAGS


def load_image(file_location):
    image = task.load_image(file_location)
    return task.prepare_image(image)

def load_shape(file_location):
    image = task.load_image(file_location)
    return tf.shape(image)[:-1]


def reshape_and_save((prob, shape, file_name)):
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
    resized_images = tf.map_fn(model.prepare_image, images)

    score = model.nldf(resized_images)

    prob = tf.nn.sigmoid(score)
    
    result = tf.map_fn(reshape_and_save, (images, shapes, output_locations), dtype=tf.string)


if __name__ == "__main__":
    tf.app.run()
