from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import input as ci
import main
import config as cf
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', cf.validate_save_data_dir,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', cf.train_save_data_dir,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f(%i/%i)' % (datetime.now(), precision,true_count,total_sample_count))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(data_dir, save_dir):
    with tf.Graph().as_default():
        images, labels = main.inputs(data_dir)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = main.inference(images)
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            main.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.contrib.deprecated.merge_all_summaries()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(cf.validate_save_data_dir,
                                               graph_def=graph_def)
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            print('evaled once')
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def validate(data_dir=cf.validate_data_dir, save_dir=cf.validate_save_data_dir):
    evaluate(data_dir, save_dir)


validate()
