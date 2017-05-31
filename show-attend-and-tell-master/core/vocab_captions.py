#coding=UTF-8

from collections import Counter

import numpy as np
import os

def _process_caption_data(caption_file, max_length):
    caption_word = list()

    with open(caption_file) as f:
        for line in f:
            if len(line.replace(' ',''))<=max_length:
                caption_word.append(line)
            else:
                caption_word.append(line[:max_length])
    
    return caption_word

def _build_vocab(captions, threshold=1):
    counter = Counter()
    max_len = 0
    for caption in captions:
        words = caption.replace('\n','').split(" ") 
        for w in words:
            counter[w] +=1
        
        if len(caption.split(" ")) > max_len:
            max_len = len(caption.split(" "))

    vocab = [word for word in counter if counter[word] >= threshold]
    print ('Filtered %d words to %d words with word count threshold %d.' % (len(counter), len(vocab), threshold))

    word_to_idx = {u'<NULL>': 0, u'<START>': 1, u'<END>': 2}
    idx = 3
    for word in vocab:
        word_to_idx[word] = idx
        idx += 1
    print("Max length of caption: ", max_len)
    return word_to_idx

def _build_caption_vector(inputcaptions, word_to_idx, max_length):
    n_examples = len(inputcaptions)
    captions = np.ndarray((n_examples,max_length+2)).astype(np.int32)   

    for caption in inputcaptions:
        words = caption.split(" ") 
        cap_vec = []
        cap_vec.append(word_to_idx['<START>'])
        for word in words:
            if word in word_to_idx:
                cap_vec.append(word_to_idx[word])
        cap_vec.append(word_to_idx['<END>'])
        
        # pad short caption with the special null token '<NULL>' to make it fixed-size vector
        if len(cap_vec) < (max_length + 2):
            for j in range(max_length + 2 - len(cap_vec)):
                cap_vec.append(word_to_idx['<NULL>']) 
        
        captions = np.asarray(cap_vec)
    print("Finished building caption vectors")
    return captions
    



