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
                                        dim_hidden=1024, n_time_step=21, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=True)

    solver = CaptioningSolver(model, data, test_data, n_epochs=20000000000, batch_size=8, update_rule='adam',
                                          learning_rate=0.001, print_every=10, save_every=1,
                              pretrained_model=None,#pretrained_model='model/model-10'
                              model_path='model/', test_model='model/model-10',
                           log_path='log/',max_len=22,result_path='result/')
    #训练用
    solver.train()
    #测试用
    solver.test()

if __name__ == "__main__":
    main()
