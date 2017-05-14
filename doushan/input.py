import h5py
import tensorflow as tf
#将训练数据作为不变变量读入
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


with tf.device('/gpu:0'):
    f=h5py.File('fc1.h5','r')
    training_data =f['train_set']

    with tf.Session() as sess:
     data_initializer = tf.placeholder(dtype=training_data.dtype,shape=training_data.shape)
    input_data = tf.Variable(data_initializer, trainable=False, collections=[])
    sess.run(input_data.initializer,feed_dict={data_initializer: training_data})
    f.close()