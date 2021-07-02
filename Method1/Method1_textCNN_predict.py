# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 02:45:25 2021

@author: T90
"""

import numpy as np
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def get_category(num):
    if(num==0):
        return ['自然人']
    elif(num==1):
        return ['法人']
    elif(num==2):
        return ['其他组织']
    elif(num==3):
        return ['自然人','法人']

def get_stopword():
    with open("Method1_textCNN/hit_stopwords.txt", "r", encoding='UTF-8') as f:  # 打开文件
        data = f.read()  # 读取文件
        #print(data)
    return data

def stopword(str_list,stopwords):
    res = []
    for i in str_list:
        if(i not in stopwords):
            res.append(i)
    return res    



def predict(ss):
    vocab = np.load('Method1_textCNN/vocab.npy',allow_pickle=True).item()
    str_list = jieba.cut(ss)
    words = stopword(str_list,get_stopword())
    words_id = []
    for i in words:
        if(i in vocab):
            words_id.append(vocab[i])
        else:
            words_id.append(0)
    words_padded=pad_sequences([words_id,words_id],maxlen=50)
    model=load_model('Method1_textCNN/model_m1_textCNN.h5')
    result = model.predict(words_padded)
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    return get_category(int(y_predict[0]))

        
