from datetime import datetime
import os.path
import time
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input as ci
import config
import main

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train(data_dir):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        # Get images and labels for CIFAR-10.

        images, labels = main.distorted_inputs(data_dir)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = main.inference(images)
        # Calculate loss.
        loss = main.loss(logits, labels)
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = main.train(loss, global_step)
        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.contrib.deprecated.merge_all_summaries()
        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)


        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


train('./train')
