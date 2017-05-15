import h5py
import numpy as np
import tensorflow as tf

#将训练数据作为不变变量读入

#with tf.device('/gpu:0'):
f=h5py.File('fc1_new.h5','r')
training_data =np.array(f['train_set'],dtype=np.int32)
training_data =training_data.transpose()
print (training_data.dtype)
print (training_data.shape)
with tf.Session() as sess:
    data_initializer = tf.placeholder(dtype=training_data.dtype,shape=training_data.shape)
    input_data = tf.Variable(data_initializer, trainable=False, collections=[])
    sess.run(input_data.initializer,feed_dict={data_initializer: training_data})
f.close()