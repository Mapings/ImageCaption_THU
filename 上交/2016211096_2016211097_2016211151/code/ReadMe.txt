
本程序需在TensorFlow+Python3环境中运行

#################################################### 模型1：CNN+LSTM基础模型  ########################################################

参考Google的Im2txt源码，源码地址: https://github.com/tensorflow/models/tree/master/im2txt

## 文件夹路径：./im2txt

## Install Required Packages

	* Bazel (instructions)
	
	* TensorFlow 1.0 or greater (instructions)
	
	* NumPy (instructions)
	
	* Natural Language Toolkit (NLTK):
	　·　First install NLTK (instructions)
	　·　Then install the NLTK data (instructions)
	
## Prepare the Training Data

	在./im2txt/data下完善放置以下数据
	
	（1）训练集图片特征的.h5矩阵，要求行是样本数，列是样本的特征；
	
	（2）训练集与图片相对应的中文Caption的txt文件，每一行表示一个样本，以\n隔开；
	
	运行caption_to_id_lac.py，生成input_seq, target_seq, mask和word_count_all.txt文件，用于后续训练

## Run training

	* 在./im2txt/configuration.py下设置必要的参数，对我们来说，主要的参数有：
	
		(1) self.vocab_size = 2000        	# 必须大于字典长度
	
		(2) self.batch_size = 32 

		(3) self.embedding_size = 4096    	# 必须等于图片特征维度

		(4) self.num_lstm_units = 512

		(5) self.initial_learning_rate = 2.0

		(6) self.learning_rate_decay_factor = 0.5
	
		(7) self.max_checkpoints_to_keep = 5
		
	* 在 ./im2txt/train_pre_load.py中设置迭代轮数：

      tf.flags.DEFINE_integer("num_epochs", 20000000000000,".")，即train多少轮，每一轮将遍历所有数据
	  
	* run ./im2txt/train.py
	
	  你将在./im2txt/my_model下发现一些checkpoint文件，形如：my_model/model.ckpt-399999.data-00000-of-00001
	  
## Run inference

	* 在./im2txt/data下放置Infer集图片特征的.h5矩阵，要求行是样本数，列是样本的特征；

	* 在./run_inference.py中设置checkpoint文件

      tf.flags.DEFINE_string("checkpoint_path", "my_model/model.ckpt-399999", "Model checkpoint file or directory containing a model checkpoint file.")

	* 运行./im2txt/run_inference.py
	
	你将在results文件夹下看到你的生成文件
	
	
############################################# 模型2：show attend and tell ################################################### 

参考Show-attend-and-tell，源码地址: https://github.com/yunjey/show-attend-and-tell

一、数据准备

	在'./data'文件夹下准备以下数据

		1.特征文件：‘image_vgg19_block5_pool_feature.h5’文件

		2.caption文件：train.txt,valid.txt

		3.在matlab环境下，运行‘./data’文件夹下的write_train_set_vgg19.m程序，生成train_valid_wordslac.txt以及vgg19_new.h5文件

		4.train_valid_wordslac.txt：共8000行caption，前8000行为训练集caption，后1000行为验证集caption

		vgg19_new.h5:训练集数据维数为9000*49*512，测试集数据维数为1000*49*512
	
二、Python工具包支持

    运行本程序，需要安装如下Python工具包：

    numpy,matplotlib,scipy,scikit-image,hickle,Pillow

三、训练模型

	1.没有之前训练的参数

		在train.py文件中

			1）函数CaptioningSolver的参数pretrained_model=None；

			2）激活solver.train()；注释掉solver.test()；

			3）运行train.py
          
	2.有之前训练的参数

		在train.py文件中，

			1）将pretrained_model模型参数文件置于model_path='model/'中

			2) 函数CaptioningSolver的参数pretrained_model='model/model-10',其中model-10为a)中放置的模型参数文件名

			3）激活solver.train()；注释掉solver.test()；

			4）运行train.py

四、测试模型：

	在train.py文件中

		1.将test_model模型参数文件置于model_path='model/'中

		2. 函数CaptioningSolver的参数test_model='model/model-10',其中model-10为a)中放置的模型参数文件名

		3.激活solver.test()；注释掉solver.train()；

		4.运行train.py
  
五、训练及测试结果

	1.训练生成的model位于model_path='model/'

	2.测试结果位于result_path='result/'
