#coding=UTF-8

from vocabfunction import word_to_id
from vocabfunction import _create_vocab

import json

#Read original data
file_train = open('train.txt','r')
captions = list()
for line in file_train.readlines():
	line = line.strip('\n')
	line = line.replace(u'\ufeff','')
	# print(line.isdigit())
	if line.isdigit():
		continue
	captions.append(line)

open('train_captions.txt', 'w').write('%s' % '\n'.join(captions)) 
file_train.close()

# make a vocabulary_id
vocab, unk_id = _create_vocab(captions)

print(unk_id)

with open('train_vocab.json', 'w') as f:
    json.dump(vocab, f, ensure_ascii=False)

#captions to words
caption_word = list()
for s in captions:
	s = list(s)
	caption_word.append(s)

thefile = open('train_caption_word.txt', 'w')
for item in caption_word:
	thefile.write("%s\n" % item)

#打印出一堆数字，啥意思

#change captions to id
# caption_id = list()
id = list()
for line in caption_word:
	# s = list()
	s = [5000]
	for word in line:
		caption_id = word_to_id(vocab, unk_id, word)
		s.append(caption_id)
	s.append(5000)
	id.append(s)

thefile = open('train_caption_id.txt', 'w')
for item in id:
	thefile.write("%s\n" % item)





