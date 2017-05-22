# ImageCaption_THU

alphabet_hanzi.py: used for set up a dictionary of Chinese characters from the training image_captions

train_new.txt: results of alphabet_hanzi.py

#----------------------------------------20170514---------------------------------------#
初步的train方法：
  cd ./im2txt
  python train_preload.py

待完成：
  @窦珊 在 ./im2txt/input_data.py下完善代码
  
  @汪洁 在 ./im2txt/input_data.py下完善代码
  
 #2017.5.16   input_seq, target_seq, mask 已完善；
  至于：#seq_embeddings 是什么意思？ 
  
  @all，写一个infer脚本，测试程序的正确性

#--------------------------------------20170522-----------------------------------------------#
train的方法：
    1.将fc1_new.h5放在./im2txt/data目录下 （如果没有这个文件，可以找窦珊拷贝）
    2.在./im2txt/configuration.py下设置必要的参数，对我们来说，主要的参数有：
    （1）self.vocab_size = 2000
    （2）self.batch_size = 32  
    （3）self.embedding_size = 4096
    （4）self.num_lstm_units = 512
    （5）self.initial_learning_rate = 2.0
    （6）self.learning_rate_decay_factor = 0.5
    （7）self.max_checkpoints_to_keep = 5
    3. 将target_seq.txt, input_seq.txt, mask.txt放在./im2txt/data文件夹下
    4. 在 ./im2txt/train_pre_load.py中设置参数：tf.flags.DEFINE_integer("num_epochs", 20000000000000,".")，即train多少轮
       运行./im2txt/train_pre_load.py
       你将在./im2txt/my_model下发现一些checkpoint文件
infer的方法：
    1. 关闭train的过程（目前train和infer不能同时运行，如果发现问题解决方法，一定微信告诉@马平烁）
    2. 将image_vgg19_fc1_feature.h5文件放在./im2txt/data文件夹下
       并且在./run_inference中设置
                    tf.flags.DEFINE_string("input_category", "train_set",                               # or validation_set
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
       即选择测试的文件是train_set还是validation_set
    3. 在./run_inference中设置checkpoint文件
                    tf.flags.DEFINE_string("checkpoint_path", "my_model2/model.ckpt-399999",       #set your checkpoint to load here
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
    4. 将word_count_all.txt放在./im2txt/data目录下
    5. 运行./im2txt/run_inference.py
