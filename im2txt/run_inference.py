#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf
import sys
sys.path.append("../")
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

import numpy as np
import h5py
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "my_model/model-word-9000-fc1-cut.ckpt-199999",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "data/wordlac_count_all.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "data/image_vgg19_fc1_feature.h5",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("input_category", "test_set",                       # or validation_set
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("Results_dir", "./Results",
                       "Directory for saving generative captions.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  # Create results directory.
  Results_dir = FLAGS.Results_dir
  if not tf.gfile.IsDirectory(Results_dir):
    tf.logging.info("Creating results directory: %s", Results_dir)
    tf.gfile.MakeDirs(Results_dir)

  g = tf.Graph()

  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  print(len(vocab.vocab))

  with tf.Session(graph=g) as sess:

    # Load the model from checkpoint.
    restore_fn(sess)
    # Load the model from checkpoint.
    #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #  print(i.name)
    #  if i.name == "lstm/basic_lstm_cell/weights:0":
    #    print(sess.run('lstm/basic_lstm_cell/weights:0'))

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    # 将infer数据作为不变变量读入
    f = h5py.File(FLAGS.input_files, 'r')
    infer_data = np.array(f[FLAGS.input_category], dtype=np.float32)

    image_embeddings = infer_data
    f.close()

    submit = []
    for image_idx in range(1000):                              #只测试1000个
      image_embedding = image_embeddings[image_idx]            # A float32 np.array with shape [embedding_size]
      captions = generator.beam_search(sess, image_embedding)  
      picture_id = str(image_idx+9000)                         #这里的image_idx可以加9000用于和图片标号对应
      # print(picture_id)
      select_first = []
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        if len(vocab.vocab)<3000:                              # 分字
          sentence = " ".join(sentence)           
          caption = "%s %s" % (picture_id, sentence) 
        else:                                                  # 分词，词之间直接相连
          sentence = "".join(sentence)         
          sentence = str(list(sentence))  
          s = sentence[1:-1]               
          s = s.replace(',','')            
          s = s.replace("'",'')            
          caption = "%s %s" % (picture_id, s)
        
        select_first.append(caption)
      submit.append(select_first[0])
      print(select_first[0])
    with codecs.open('Results/word-9000-fc1-cut-big-batch-199999-beam-4.txt', 'w', encoding='utf-8') as file:
      submit_str = '\n'.join(submit)
      file.write(submit_str)

if __name__ == "__main__":
  tf.app.run()