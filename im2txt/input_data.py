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
    self.read_image_embeddings(self.train_dir)
    self.read_seqs(self.train_dir)

  def read_image_embeddings(self, train_dir):
 
	#将训练数据作为不变变量读入
	f=h5py.File('fc1_new.h5','r')
	training_data =np.array(f['train_set'],dtype=np.int32)
	training_data =training_data.transpose()

	with tf.Session() as sess:
		data_initializer = tf.placeholder(dtype=training_data.dtype,shape=training_data.shape)
		self.image_embeddings = tf.Variable(data_initializer, trainable=False, collections=[])
		sess.run(self.image_embeddings.initializer,feed_dict={data_initializer: training_data})
	f.close()
    

  def read_seqs(self, train_dir):
    
    #seq_embeddings 是什么意思？ 
    self.seq_embeddings = np.random.randint(11, size=(700, 50), dtype=np.int32)

    #读入已经处理好的input_seq, target_seq以及mask 
    f_input = open('input_seq.txt','r')
    self.input_seqs = np.array(f_input,dtype=np.int32)
    f_target = open('target_seq.txt','r')
    self.target_seqs = np.array(f_target,dtype=np.int32)
    f_mask = open('mask.txt','r')
    self.mask = np.array(f_mask,dtype=np.int32)
	
    f_input.close()
    f_target.close()
    f_mask.close()
