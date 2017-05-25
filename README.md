# ImageCaption_THU

# -------------------------------------- 20170522: train和infer的教程 -----------------------------------------------#


##train的方法：##


    1.将fc1_new.h5 或者fc2_new.h5 放在./im2txt/data目录下
    
    2.在./im2txt/configuration.py下设置必要的参数，对我们来说，主要的参数有：
    
		(1) self.vocab_size = 2000
    
		(2) self.batch_size = 32  
    
		(3) self.embedding_size = 4096
    
		(4) self.num_lstm_units = 512
    
		(5) self.initial_learning_rate = 2.0
    
		(6) self.learning_rate_decay_factor = 0.5
    
		(7) self.max_checkpoints_to_keep = 5
    
    3. 将对应的target_seq.txt, input_seq.txt, mask.txt放在./im2txt/data文件夹下
    
    4. 在 ./im2txt/train_pre_load.py中设置迭代轮数：
    	
       tf.flags.DEFINE_integer("num_epochs", 20000000000000,".")，即train多少轮，每一轮将遍历所有数据
	
       运行./im2txt/train_pre_load.py
       
       你将在./im2txt/my_model下发现一些checkpoint文件
       

##infer的方法：##


    1. 关闭train的过程（目前train和infer不能同时运行，如果发现问题解决方法，一定微信告诉@马平烁）
    
    2. 将image_vgg19_fc1_feature.h5 或者image_vgg19_fc2_feature.h5文件放在./im2txt/data文件夹下
    
       并且在./run_inference中设置
       
                    tf.flags.DEFINE_string("input_category", "train_set",                   # or validation_set or test_set
		    
                       "File pattern or comma-separated list of file patterns "
                       
                       "of image files.")
                       
       即选择测试的文件是train_set还是validation_set还是test_set
      
    3. 在./run_inference中设置checkpoint文件
    
    	tf.flags.DEFINE_string("checkpoint_path", "my_model/model.ckpt-399999",       #set your checkpoint to load here
		
			"Model checkpoint file or directory containing a "
			
			 "model checkpoint file.")
                       
    4. 将对应的word_count_all.txt放在./im2txt/data目录下
    
    5. 运行./im2txt/run_inference.py
    
#--------------------------------------20170524：关于不同训练字典、不同参数如何训练和测试----------------------------------------------#

##创建了两个文件夹，包含新的训练字典：##

   1. ./wangjie/lac_2, 使用中文分词软件 THULAC，min_word_count = 2
   
   	### training
    
    	· 在./im2txt/configuration.py下设置必要的参数：(1) self.vocab_size = 6000
	
	· 将该文件夹下的target_seq.txt, input_seq.txt, mask.txt放在./im2txt/data文件夹下
	
	· train it 
	
	### inference
	
	· 将wordlac_count_all.txt放在./im2txt/data目录下。这里一定要注意，使用所有word的频数统计的txt，否则会生成unk_id
	
	· run_inference.py里面做相应修改：
	tf.flags.DEFINE_string("vocab_file", "data/wordlac_count_all.txt", "Text file containing the vocabulary.")
	
	· train it 
	
	
   2. ./wangjie/word_3, 分割出一个一个中文汉字和一个一个英文单词，min_word_count = 3
   
   	### training
    
    	· 在./im2txt/configuration.py下设置必要的参数：(1) self.vocab_size = 2000
	
	· 其他同上
	
	#--------------------------------------20170525:生成用于提交测试的txt文件-----------------------------------------------#

	run_inference.py 即可
	
	生成的submit.txt文件在工作目录下新建的文件夹Results中
	
	
