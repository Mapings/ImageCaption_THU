#coding=UTF-8

from vocabfunction import Vocabulary
from vocabfunction import _create_vocab
# from vocabfunction import caption_to_input
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

open('captions.txt', 'w').write('%s' % '\n'.join(captions)) 
file_train.close()

# split Chinese and english word by thulac
thu1 = thulac.thulac(seg_only=True)
thu1.cut_f("captions.txt", "train_wordslac.txt")

# add special start and end words
file_trainwords = open('train_wordslac.txt','r')
caption_word = list()
for line in file_trainwords.readlines():
	start = ["<START>"];
	stop = "<END>";
	start.extend(line.split())
	start.append(stop)
	caption_word.append(start)

open('train_captions.txt', 'w').write('%s' % '\n'.join(str(v) for v in caption_word)) 
file_trainwords.close()


# make a vocabulary_id; create word_counts.txt
vocab, unk_id = _create_vocab(caption_word)

# print(unk_id)
# print(vocab._unk_id)







