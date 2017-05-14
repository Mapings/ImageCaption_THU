# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import sys
sys.path.append("../")
from im2txt import configuration
from im2txt import show_and_tell_model
import time
import input_data

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "./test_mps_train_dir",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("num_epochs", 20,
                        ".")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
  # assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.train_dir, "--train_dir is required"

  model_config = configuration.ModelConfig()
  # model_config.input_file_pattern = FLAGS.input_file_pattern
  # model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # 从文件读取数据
  # data_sets = input_data.read_data_sets(FLAGS.train_dir)

  # for 测试
  data_sets = input_data.read_data_sets(FLAGS.train_dir)

  # Build the TensorFlow graph.
  g = tf.Graph()
  with g.as_default():
    with tf.name_scope('input'):
      # Input data
      image_embeddings_initializer = tf.placeholder(
          dtype=data_sets.image_embeddings.dtype,
          shape=data_sets.image_embeddings.shape)
      seq_embeddings_initializer = tf.placeholder(
          dtype=data_sets.seq_embeddings.dtype,
          shape=data_sets.seq_embeddings.shape)
      target_seqs_initializer = tf.placeholder(
          dtype=data_sets.target_seqs.dtype,
          shape=data_sets.target_seqs.shape)
      input_mask_initializer = tf.placeholder(
          dtype=data_sets.input_mask.dtype,
          shape=data_sets.input_mask.shape)

      input_image_embeddings = tf.Variable(
          image_embeddings_initializer, trainable=False, collections=[])
      input_seq_embeddings = tf.Variable(
          seq_embeddings_initializer, trainable=False, collections=[])
      input_target_seqs = tf.Variable(
          target_seqs_initializer, trainable=False, collections=[])
      input_input_mask = tf.Variable(
          input_mask_initializer, trainable=False, collections=[])

      image_embeddings_slice, seq_embeddings_slice, target_seqs_slice, input_mask_slice = tf.train.slice_input_producer(
          [input_image_embeddings, input_seq_embeddings, input_target_seqs, input_input_mask], num_epochs=FLAGS.num_epochs)
      image_embeddings, seq_embeddings, target_seqs, input_mask = tf.train.batch(
          [image_embeddings_slice, seq_embeddings_slice, target_seqs_slice, input_mask_slice], batch_size=model_config.batch_size)
    print(image_embeddings)
    print(seq_embeddings)
    print(target_seqs)
    print(input_mask)

    # Build the model.
    model = show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", image_embeddings=image_embeddings, seq_embeddings=seq_embeddings, target_seqs=target_seqs, input_mask=input_mask)
    model.build()

    # Set up the learning rate.
    learning_rate_decay_fn = None
    if FLAGS.train_inception:
        learning_rate = tf.constant(training_config.train_inception_learning_rate)
    else:
        learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn
    print(model.global_step)
    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

    # Create the op for initializing variables.
    init_op = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer(),)

    #init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    sess.run(init_op)

    #feed_dict = {image_embeddings_initializer: data_sets.image_embeddings}
    #print(feed_dict)
    sess.run(input_image_embeddings.initializer,
           feed_dict={image_embeddings_initializer: data_sets.image_embeddings})
    sess.run(input_seq_embeddings.initializer,
           feed_dict={seq_embeddings_initializer: data_sets.seq_embeddings})
    sess.run(input_target_seqs.initializer,
           feed_dict={target_seqs_initializer: data_sets.target_seqs})
    sess.run(input_input_mask.initializer,
           feed_dict={input_mask_initializer: data_sets.input_mask})

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print("here")
    # Run training.

    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        # Run one step of the model.
        _, loss_value = sess.run([train_op, model.total_loss])
        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 1 == 0:
          # Print status to stdout.
          print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                           duration))
          # Update the events file.
          #summary_str = sess.run(summary_op)
          #summary_writer.add_summary(summary_str, step)
          step += 1

      # Save a checkpoint periodically.
        if (step + 1) % 1000 == 0:
          print('Saving')
          saver.save(sess, FLAGS.train_dir, global_step=step)

          step += 1
    except tf.errors.OutOfRangeError:
      print('Saving')
      saver.save(sess, FLAGS.train_dir, global_step=step)
      print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
    finally:
    # When done, ask the threads to stop.
      coord.request_stop()

      # Wait for threads to finish.
    coord.join(threads)
    
    '''
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        init_op = init_op,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver)
    '''

if __name__ == "__main__":
  tf.app.run()
