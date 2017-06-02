# show attend and tell

#-------------------------------------- 20170528: 请添加修改 -----------------------------------------------#

@汪洁：请在./show-attend-and-tell-master/core/utils.py中添加相应字典输入代码

已经添加，尚未测试

data需要的captions， word_to_idx 也已经生成，在./data文件夹下。需要结合上述添加的代码测试一下。
	
#-------------------------------------- 20170602: 请添加修改 -----------------------------------------------#	

1.训练模式：

a)没有之前训练的参数

在train.py文件中，令函数CaptioningSolver的参数pretrained_model=None,mode='train'
          
b)有之前训练的参数

在train.py文件中，令函数CaptioningSolver的参数pretrained_model='model/~~.cpkt',mode='train'

2.测试模式：在train.py文件中，令函数CaptioningSolver的参数pretrained_model='model/~~.cpkt',mode='test'
  
3.将pretrained_model文件置于model_path='model/'中（训练生成的model也在相同位置）

4.生成的测试文件在result_path='result/'中
