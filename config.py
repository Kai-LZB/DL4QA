#! -*- coding:utf-8 -*-
'''
@author: Kai Zheng
'''

class TrainConfig(object):
    WORD_EMBEDDING_DIM = 50
    IDF_NUM = 3
    TRAIN_EPOCHS = 50
    W2V_MIN_COUNT = 5
    LOSS_FUNC = 'binary_crossentropy'
    OPTIMIZER = 'adam'