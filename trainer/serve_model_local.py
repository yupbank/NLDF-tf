import os
import tensorflow as tf
from tensorflow.contrib.saved_model.python.saved_model import reader  # pylint: disable=no-name-in-module
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils  # pylint: disable=no-name-in-module

tf.app.flags.DEFINE_string('model_location', '/tmp/nldf-network/exported',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_string('image_location', '/tmp/img.jpg',
                           "Needs to provide the input image location")
tf.app.flags.DEFINE_string('output_location', '/tmp/vector',
                           "Needs to provide the location to save")
FLAGS = tf.app.flags.FLAGS


def load_meta_graph(model_path, tags, graph, signature_def_key=None):
    saved_model = reader.read_saved_model(model_path)
    the_meta_graph = None
    for meta_graph_def in saved_model.meta_graphs:
        if sorted(meta_graph_def.meta_info_def.tags) == sorted(tags):
            the_meta_graph = meta_graph_def
    signature_def_key = signature_def_key or tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    try:
        signature_def = signature_def_utils.get_signature_def_by_key(
            the_meta_graph,
            signature_def_key)
    except ValueError as ex:  # pylint: disable=except-too-long
        try:
            formatted_key = 'default_input_alternative:{}'.format(signature_def_key)
            signature_def = signature_def_utils.get_signature_def_by_key(the_meta_graph, formatted_key)
        except ValueError:
            raise ValueError(
                'Got signature_def_key "{}". Available signatures are {}. '
                'Original error:\n{}'.format(signature_def_key, list(the_meta_graph.signature_def), ex)
            )
    input_names = {k: v.name for k, v in signature_def.inputs.items()}
    output_names = {k: v.name for k, v in signature_def.outputs.items()}
    feed_tensors = {k: graph.get_tensor_by_name(v)
                    for k, v in input_names.items()}
    fetch_tensors = {k: graph.get_tensor_by_name(v)
                     for k, v in output_names.items()}
    return feed_tensors['image_locations'], \
           feed_tensors['output_locations'], \
           fetch_tensors['feature_map']


def load_model(model_path):
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    tags = [tf.saved_model.tag_constants.SERVING]
    tf.saved_model.loader.load(sess, tags, model_path)
    image_location, vector_location, fetch  = load_meta_graph(model_path, tags, graph)
    def _inference(file_names, out_files):
        return sess.run(fetch, feed_dict={image_location: file_names, vector_location: out_files})
    return _inference


def main(_=None):
    model_path = os.path.join(FLAGS.model_location, str(FLAGS.model_version))
    do_inference = load_model(model_path)
    import ipdb; ipdb.set_trace()
    print(
        'successly get saliency map:',
        FLAGS.image_location,
        'at location:',
        FLAGS.output_location,
        'with status:',
        do_inference([FLAGS.image_location], [FLAGS.output_location])
    )


if __name__ == '__main__':
    tf.app.run()
