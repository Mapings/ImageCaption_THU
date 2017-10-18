from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import h5py

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
    self.image_embeddings = None
    self.input_seqs = None
    self.target_seqs = None
    self.input_mask = None
    self.read_image_embeddings(self.train_dir)
    self.read_seqs(self.train_dir)

  def read_image_embeddings(self, data_dir):
 
    # 将训练数据作为不变变量读入
    f = h5py.File(data_dir+'/fc1_9000_new.h5','r')
    training_data = np.array(f['train_set'],dtype=np.float32)
    training_data = training_data.transpose()
    self.image_embeddings = training_data
    f.close()

  def read_seqs(self, data_dir):

    #self.seq_embeddings = np.random.randint(11, size=(700, 50), dtype=np.int32)

    # 读入已经处理好的input_seq, target_seq以及mask
    f_input = open(data_dir + '/input_seq.txt','r')
    input_seqs = []
    for line in f_input:
        input_seqs.append(eval(line))
    self.input_seqs = np.array(input_seqs, dtype=np.int32)

    f_target = open(data_dir + '/target_seq.txt','r')
    target_seqs = []
    for line in f_target:
        target_seqs.append(eval(line))
    self.target_seqs = np.array(target_seqs, dtype=np.int32)

    f_mask = open(data_dir + '/mask.txt','r')
    input_mask = []
    for line in f_mask:
        input_mask.append(eval(line))
    self.input_mask = np.array(input_mask, dtype=np.int32)
	
    f_input.close()
    f_target.close()
    f_mask.close()
