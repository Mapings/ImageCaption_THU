#coding=UTF-8

from vocabfunction import Vocabulary
from vocabfunction import _create_vocab
from vocabfunction import caption_to_input
import thulac

#Read original data
file_train = open('train.txt','r')
captions = list()
for line in file_train.readlines():
	line = line.strip('\n')
	line = line.replace(u'\ufeff','')
	# print(line.isdigit())
	if line.isdigit():
		continue
	line = line.upper()
	# line = line.replace("1","一")
	# line = line.replace("2","二")
	# line = line.replace("3","三")
	# line = line.replace("4","四")
	# line = line.replace("5","五")
	# line = line.replace("6","六")
	line = line.replace(",","，")
	captions.append(line)

# open('train_captions.txt', 'w').write('%s' % '\n'.join(captions)) 
# file_train.close()

# split Chinese and english word by word & add special start and end words
caption_word = list()
for s in captions:
	start = ["$"];
	stop = "%";
	s = list(s)
	index = []
	english = ''
	for i in range(len(s)):
		if s[i].isupper():
			english +=s[i]
			index.append(i)
	if index:
		temp = s[:index[0]]
		temp.append(english)
		temp.extend(s[index[len(index)-1]+1:])
	else:
		temp = s
	start.extend(temp)
	start.append(stop)
	caption_word.append(start)

# thefile = open('train_caption_word.txt', 'w')
# for item in caption_word:
# 	thefile.write("%s\n" % item)

# make a vocabulary_id; create word_counts.txt
vocab, unk_id = _create_vocab(caption_word)

# print(unk_id)
# print(vocab._unk_id)


#change captions to id
id = list()
seqLen = 22;
for line in caption_word:
	s = []
	if len(line)<=seqLen:
		for word in line[:seqLen]:
			caption_id = Vocabulary.word_to_id(vocab._vocab, unk_id, word)
			s.append(caption_id)
	else:
		line = line[:(seqLen-1)]
		line.extend("%")
		for word in line[:seqLen]:
			caption_id = Vocabulary.word_to_id(vocab._vocab, unk_id, word)
			s.append(caption_id)
	id.append(s)

# thefile = open('train_caption_id.txt', 'w')
# for item in id:
# 	thefile.write("%s\n" % item)

input_seq, target_seq, mask = caption_to_input(seqLen, id)
thefile = open('input_seq.txt', 'w')
for item in input_seq:
	thefile.write("%s\n" % item)
thefile = open('target_seq.txt', 'w')
for item in target_seq:
	thefile.write("%s\n" % item)
thefile = open('mask.txt', 'w')
for item in mask:
	thefile.write("%s\n" % item)






