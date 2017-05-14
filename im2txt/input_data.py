from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

"""Input Data Here
  Outputs:
      image_embeddings
      seq_embeddings
      target_seqs(training and eval only)
      input_mask(training and eval only)
      
  使用方法：
      import input_data
      data_sets = input_data.read_data_sets(FLAGS.train_dir) 读取以上四个内容，用以训练
"""

class read_data_sets(object):
  def __init__(self, train_dir):

    self.train_dir = train_dir
    self.read_image_embeddings(self.train_dir)
    self.read_seqs(self.train_dir)

  def read_image_embeddings(self, train_dir):
    #TODO
    #删掉下面两句话，并且换成你自己的代码
    self.image_embeddings = np.random.rand(700, 512)
    self.image_embeddings = self.image_embeddings.astype(np.float32)

  def read_seqs(self, train_dir):
    #TODO
    #删掉下面三句话，并且换成你自己的代码
    self.seq_embeddings = np.random.randint(11, size=(700, 50), dtype=np.int32)
    self.target_seqs = np.random.randint(11, size=(700, 50), dtype=np.int32)
    self.input_mask = np.ones((700, 50), dtype=np.int32)