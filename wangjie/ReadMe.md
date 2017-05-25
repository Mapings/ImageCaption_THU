Caption 处理文件说明如下：

原始数据：
train.txt

代码：
1. vocabfunction.py, 包含三个函数，一个是建立字典（汉字 - id 对应关系的字典），一个是汉字-id的查询，一个是从caption输出input_seq, target_seq, mask。
2. caption_to_id.py, 有如下功能：读入train.txt的captions文件，生成vocabulary；为每一句caption生成对应的id编码的list文件（该list文件的每一句caption都是一个list。统一caption的最大长度为seqLen(12), 每句caption的起始为相同的一个字符，其id设为字典长度+10；每句话的结束也设置为相同的字符，其id设为字典长度+11。关于target_seq是否统一为以结束字符结束，可以通过调整 seqLen和seqlength的大小调整。当前代码下，以相同字符结束。

生成的文件：

主要：

input_seq； target_seq； mask； train_vocab.txt(min_word_count = 4), 字典（记得重新上传更新之后的，1所谓字典起始的id；0已经作为padding word了）。

过程中生成的文件：

word_count_all.txt(设置min_word_count = 1，得到train caption中所有字的词频统计结果）；

train_captions.txt,原始标注数据去除图片数字标号的中文描述；

train_caption_word是将train_captions分割成一个一个字，用于最后生成train_caption_id.txt。


#----------------------------------------2017-05-24-------------------------------------------------#


新建两个文件夹：
1. lac_2 
中文分词后，重新生成的caption输入文件, 使用THULAC中文分词工具，min_word_count = 2

2. word_3
将每个中文汉字和每个英文单词作为字典的元素，重新生成的caption的输入文件，min_word_count = 3

#----------------------------------------2017-05-25-------------------------------------------------#

进一步的工作：1. 把UNK这个问题再处理一下；2.fc2+分词；3.fc1后面连一个全连接映射到512； 4.直接fc1+fc2成一个向量，就是4096*2
