Caption 处理文件说明如下：

原始数据：
train.txt

代码：
1. vocabfunction.py, 包含三个函数，一个是建立字典（汉字 - id 对应关系的字典），一个是汉字-id的查询，一个是从caption输出input_seq, target_seq, mask。
2. caption_to_id.py, 有如下功能：读入train.txt的captions文件，生成vocabulary；为每一句caption生成对应的id编码的list文件（该list文件的每一句caption都是一个list。统一caption的最大长度为seqLen(12), 每句caption的起始为相同的一个字符，其id设为字典长度+10；每句话的结束也设置为相同的字符，其id设为字典长度+11。关于target_seq是否统一为以结束字符结束，可以通过调整 seqLen和seqlength的大小调整。当前代码下，以相同字符结束。

生成的文件：

主要：input_seq, target_seq, mask.

过程中生成的文件：word_count_all.txt(设置min_word_count = 1，得到train caption中所有字的词频统计结果）；train_vocab.txt, 字典；train_captions.txt,原始标注数据去除图片数字标号的中文描述；train_caption_word,train_captions转化成一个一个字，用于最后生成train_caption_id.txt


