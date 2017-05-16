#coding=UTF-8

from vocabfunction import word_to_id
from vocabfunction import _create_vocab
from vocabfunction import caption_to_input

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


#change captions to id
# caption_id = list()
id = list()
seqLen = 12;
for line in caption_word:
	# s = list()
	# each caption with a specific start and stop (not in the vocabulary)
	start = [unk_id + 10];
	stop = [unk_id + 11];
	for word in line[:seqLen]:
		caption_id = word_to_id(vocab, unk_id, word)
		start.append(caption_id)
	start.append(stop)
	id.append(start)

thefile = open('train_caption_id.txt', 'w')
for item in id:
	thefile.write("%s\n" % item)

seqlength = seqLen+2;
input_seq, target_seq, mask = caption_to_input(seqlength, id)
thefile = open('input_seq.txt', 'w')
for item in input_seq:
	thefile.write("%s\n" % item)
thefile = open('target_seq.txt', 'w')
for item in target_seq:
	thefile.write("%s\n" % item)
thefile = open('mask.txt', 'w')
for item in mask:
	thefile.write("%s\n" % item)
