#coding=UTF-8

from vocab_captions import _process_caption_data
from vocab_captions import _build_vocab
from vocab_captions import _build_caption_vector

import numpy as np
import time
import os
import h5py

def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}
  
    # 读入features
    f = h5py.File(data_path+'/vgg19_new.h5','r')
    split_data = np.array(f[split],dtype=np.float32)
    data['features']=np.transpose(split_data,(2,1,0))
    if split == 'train':
        data['file_names'] = np.arange(split_data.shape[0])+1
        #image_idxs: Indices for mapping caption to image of shape（40000，） 设caption为40000个
        image_idxs = np.array(f['image_idxs'],dtype=np.int32)
        data['image_idxs'] = np.transpose(image_idxs)
    f.close()
    
    # maximum length of caption(number of word). if caption is longer than max_length, deleted.  
    max_length = 20
    # if word occurs less than word_count_threshold in training dataset, the word index is special unknown token.
    word_count_threshold = 1
    #word_to_idx: Mapping dictionary from word to index

    if split == 'train':
        train_captions = _process_caption_data(caption_file=data_path + '/train_wordslac.txt', 
                                               max_length=20)
        word_to_idx = _build_vocab(captions=train_captions, 
                                   threshold=word_count_threshold)
        data['word_to_idx'] = word_to_idx
        captions = _build_caption_vector(inputcaptions=train_captions, 
                                     word_to_idx=word_to_idx, max_length=max_length)
        data['captions'] = captions

    for k, v in data.items():
        if type(v) == np.ndarray:
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), len(v))
    end_t = time.time()
    print("Elapse time: %.2f" %(end_t - start_t))
    return data

#@汪洁，在这里加入由序列转换为句子
def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            #如果null与结束符号相同，这里应该做相应修改
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded

def sample_coco_minibatch(data, batch_size):
    data_size = data['features'].shape[0]
    mask = np.random.choice(data_size, batch_size)
    features = data['features'][mask]
    file_names = data['file_names'][mask]
    return features, file_names

def write_bleu(scores, path, epoch):
    if epoch == 0:
        file_mode = 'w'
    else:
        file_mode = 'a'
    with open(os.path.join(path, 'val.bleu.scores.txt'), file_mode) as f:
        f.write('Epoch %d\n' %(epoch+1))
        f.write('Bleu_1: %f\n' %scores['Bleu_1'])
        f.write('Bleu_2: %f\n' %scores['Bleu_2'])
        f.write('Bleu_3: %f\n' %scores['Bleu_3'])  
        f.write('Bleu_4: %f\n' %scores['Bleu_4']) 
        f.write('METEOR: %f\n' %scores['METEOR'])  
        f.write('ROUGE_L: %f\n' %scores['ROUGE_L'])  
        f.write('CIDEr: %f\n\n' %scores['CIDEr'])

def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file  

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)
