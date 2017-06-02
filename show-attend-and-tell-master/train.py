#coding=UTF-8

from solver import CaptioningSolver
from model import CaptionGenerator
from utils import load_coco_data


def main():
    # load train dataset
    data = load_coco_data(data_path='./data', split='train')
    word_to_idx = data['word_to_idx']
    # load val dataset to print out bleu scores every epoch
    test_data = load_coco_data(data_path='./data', split='test')

    model = CaptionGenerator(word_to_idx, dim_feature=[49, 512], dim_embed=512,
                                       # dim_hidden=1024, n_time_step=22, prev2out=True,
                                        dim_hidden=1024, n_time_step=21, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, test_data, n_epochs=20000000000, batch_size=8, update_rule='adam',
                                          learning_rate=0.001, print_every=10, save_every=1,
                                    pretrained_model=None, model_path='model/lstm/', test_model='model/lstm/model-10',
                              #test_or_not=True,
                              mode='train', log_path='log/',max_len=22,result_path='result/')

    solver.train()

if __name__ == "__main__":
    main()
