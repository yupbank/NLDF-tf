import tensorflow as tf
import model

image_size = 352
label_size = image_size/2

def main(_):
    with tf.Graph().as_default() as g:
        inputs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
        labels = tf.placeholder(tf.float32, [None, label_size, label_size, 1])

        prob, endpoints = model.nldf(inputs, labels)
        loss = endpoints['loss']
        tvars = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(1e-6)
        train_op = model.clip_grad(opt, loss, tvars, opt, max_grad_norm)
        accuracy = endpoints['accuracy']
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if restore_from_vgg:
                vgg_initials = endpoints['vggs']
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(vgg_model_path, vggs_initials)
                sess.run(init_fn)

            sess.run(train_op)

    saver.save(sess, 'Model/model.ckpt', global_step=n_epochs)



