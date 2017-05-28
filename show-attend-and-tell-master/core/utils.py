import numpy as np
import cPickle as pickle
import hickle
import time
import os


def load_coco_data(data_path='./data', split='train'):

    start_t = time.time()
    data = {}
  
  
  
    # 读入features
    f = h5py.File(data_path+'/vgg19_new.h5','r')
    split_data = np.array(f[split],dtype=np.float32)
    data['features']=np.transpose(split_data,(2,1,0))
    if split == 'train':
        data['file_names'] = np.arange(split_data.shape[0])+1
        #image_idxs: Indices for mapping caption to image of shape（40000，） 设caption为40000个
        image_idxs = np.array(f['image_idxs'],dtype=np.float32)
        data['image_idxs'] = np.transpose(image_idxs)
    
    
    #@汪洁在这里写入caption和'word_to_idx'，注意前面读文件已经用过f变量
    #captions: Captions of shape (400000, 17) 是句长为17
    #word_to_idx: Mapping dictionary from word to index
    if split == 'train':
        data['captions'] = pickle.load(f)
        with open(os.path.join(data_path, 'word_to_idx.pkl'), 'rb') as f:
            data['word_to_idx'] = pickle.load(f)


    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print k, type(v), v.shape, v.dtype
        else:
            print k, type(v), len(v)
    end_t = time.time()
    print "Elapse time: %.2f" %(end_t - start_t)
    f.close()
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
