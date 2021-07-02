# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:28:46 2021

@author: T90
"""
from gensim.models import word2vec
import logging
 
##训练word2vec模型
 
# 获取日志信息
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
 
# 加载分词后的文本，使用的是Text8Corpus类
 
sentences = word2vec.Text8Corpus(r'corpus.txt')
# 训练模型，部分参数如下

model = word2vec.Word2Vec(sentences, sg=1, vector_size=300,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
model.save('word2vec2.model')
print(model.wv['原告'])
