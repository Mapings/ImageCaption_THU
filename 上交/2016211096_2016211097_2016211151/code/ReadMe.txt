
����������TensorFlow+Python3����������

#################################################### ģ��1��CNN+LSTM����ģ��  ########################################################

�ο�Google��Im2txtԴ�룬Դ���ַ: https://github.com/tensorflow/models/tree/master/im2txt

## �ļ���·����./im2txt

## Install Required Packages

	* Bazel (instructions)
	
	* TensorFlow 1.0 or greater (instructions)
	
	* NumPy (instructions)
	
	* Natural Language Toolkit (NLTK):
	������First install NLTK (instructions)
	������Then install the NLTK data (instructions)
	
## Prepare the Training Data

	��./im2txt/data�����Ʒ�����������
	
	��1��ѵ����ͼƬ������.h5����Ҫ������������������������������
	
	��2��ѵ������ͼƬ���Ӧ������Caption��txt�ļ���ÿһ�б�ʾһ����������\n������
	
	����caption_to_id_lac.py������input_seq, target_seq, mask��word_count_all.txt�ļ������ں���ѵ��

## Run training

	* ��./im2txt/configuration.py�����ñ�Ҫ�Ĳ�������������˵����Ҫ�Ĳ����У�
	
		(1) self.vocab_size = 2000        	# ��������ֵ䳤��
	
		(2) self.batch_size = 32 

		(3) self.embedding_size = 4096    	# �������ͼƬ����ά��

		(4) self.num_lstm_units = 512

		(5) self.initial_learning_rate = 2.0

		(6) self.learning_rate_decay_factor = 0.5
	
		(7) self.max_checkpoints_to_keep = 5
		
	* �� ./im2txt/train_pre_load.py�����õ���������

      tf.flags.DEFINE_integer("num_epochs", 20000000000000,".")����train�����֣�ÿһ�ֽ�������������
	  
	* run ./im2txt/train.py
	
	  �㽫��./im2txt/my_model�·���һЩcheckpoint�ļ������磺my_model/model.ckpt-399999.data-00000-of-00001
	  
## Run inference

	* ��./im2txt/data�·���Infer��ͼƬ������.h5����Ҫ������������������������������

	* ��./run_inference.py������checkpoint�ļ�

      tf.flags.DEFINE_string("checkpoint_path", "my_model/model.ckpt-399999", "Model checkpoint file or directory containing a model checkpoint file.")

	* ����./im2txt/run_inference.py
	
	�㽫��results�ļ����¿�����������ļ�
	
	
############################################# ģ��2��show attend and tell ################################################### 

�ο�Show-attend-and-tell��Դ���ַ: https://github.com/yunjey/show-attend-and-tell

һ������׼��

	��'./data'�ļ�����׼����������

		1.�����ļ�����image_vgg19_block5_pool_feature.h5���ļ�

		2.caption�ļ���train.txt,valid.txt

		3.��matlab�����£����С�./data���ļ����µ�write_train_set_vgg19.m��������train_valid_wordslac.txt�Լ�vgg19_new.h5�ļ�

		4.train_valid_wordslac.txt����8000��caption��ǰ8000��Ϊѵ����caption����1000��Ϊ��֤��caption

		vgg19_new.h5:ѵ��������ά��Ϊ9000*49*512�����Լ�����ά��Ϊ1000*49*512
	
����Python���߰�֧��

    ���б�������Ҫ��װ����Python���߰���

    numpy,matplotlib,scipy,scikit-image,hickle,Pillow

����ѵ��ģ��

	1.û��֮ǰѵ���Ĳ���

		��train.py�ļ���

			1������CaptioningSolver�Ĳ���pretrained_model=None��

			2������solver.train()��ע�͵�solver.test()��

			3������train.py
          
	2.��֮ǰѵ���Ĳ���

		��train.py�ļ��У�

			1����pretrained_modelģ�Ͳ����ļ�����model_path='model/'��

			2) ����CaptioningSolver�Ĳ���pretrained_model='model/model-10',����model-10Ϊa)�з��õ�ģ�Ͳ����ļ���

			3������solver.train()��ע�͵�solver.test()��

			4������train.py

�ġ�����ģ�ͣ�

	��train.py�ļ���

		1.��test_modelģ�Ͳ����ļ�����model_path='model/'��

		2. ����CaptioningSolver�Ĳ���test_model='model/model-10',����model-10Ϊa)�з��õ�ģ�Ͳ����ļ���

		3.����solver.test()��ע�͵�solver.train()��

		4.����train.py
  
�塢ѵ�������Խ��

	1.ѵ�����ɵ�modelλ��model_path='model/'

	2.���Խ��λ��result_path='result/'
