Caption 处理文件说明如下：

原始数据：
train.txt

代码：
1. vocabfunction.py, 包含两个函数，一个是建立字典（汉字 - id 对应关系的字典），一个是汉字-id的查询。
2. caption_to_id.py, 有如下功能：读入train.txt的captions文件，生成vocabulary；为每一句caption生成对应的id编码的list文件（该list文件的每一句caption都是一个list，每句caption的起始和中止id都设置为相同的一个数，设为5000

生成的文件：

主要：train_caption_id.txt, caption的id编码

过程中生成的文件：train_vocab.txt, 字典；train_captions.txt,原始标注数据去除图片数字标号的中文描述；train_caption_word,train_captions转化成一个一个字，用于最后生成train_caption_id.txt


