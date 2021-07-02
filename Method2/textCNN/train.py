# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:30:21 2021

@author: T90
"""
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def get_stopword():
    with open("hit_stopwords.txt", "r", encoding='UTF-8') as f:  # 打开文件
        data = f.read()  # 读取文件
        #print(data)
    return data

def stopword(str_list,stopwords):
    res = []
    for i in str_list:
        if(i not in stopwords):
            res.append(i)
    return res



csv_file = 'data_train.csv'
df = pd.read_csv(csv_file, low_memory = False)#防止弹出警告
df['sentence'] = df['sentence'].astype(str)
df['label'] = df['label'].astype(str)


stopword_list = get_stopword()
cw = lambda x: list(jieba.cut(x))  #分词
st = lambda x: stopword(x,stopword_list)#停词

df['words'] = df['sentence'].apply(cw)
df['words'] = df['words'].apply(st)
tokenizer=Tokenizer()  #创建一个Tokenizer对象
#fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
tokenizer.fit_on_texts(df['words'])
vocab=tokenizer.word_index #得到每个词的编号
#存储字典
np.save('vocab.npy', vocab)

# x_train, x_test, y_train, y_test = train_test_split(df['words'], df['label'], test_size=0.1)
x_train = df['words']
y_train = df['label']
# 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
x_train_word_ids=tokenizer.texts_to_sequences(x_train)
# x_test_word_ids = tokenizer.texts_to_sequences(x_test)
#序列模式 每条样本长度不唯一，将每条样本的长度设置一个固定值
x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=20) #将超过固定值的部分截掉，不足的在最前面用0填充
# x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=20)

from gensim.models import word2vec
w2v_model=word2vec.Word2Vec.load('word2vec2.model')
# 预训练的词向量中没有出现的词用0向量表示
embedding_matrix = np.zeros((len(vocab) + 1, 300))
for word, i in vocab.items():
    try:
        embedding_vector = w2v_model.wv[str(word)]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        continue
    
from keras.layers import Input,Embedding,Conv1D,MaxPooling1D,concatenate,Flatten
from keras.layers import Dropout,Dense
import keras
from sklearn import metrics
from keras.models import Model
from keras.utils import np_utils
#构建TextCNN模型
#模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接

main_input = Input(shape=(20,), dtype='float64')
# 词嵌入（使用预训练的词向量）

embedder = Embedding(len(vocab) + 1, 300, input_length=50, weights=[embedding_matrix], trainable=False)
embed = embedder(main_input)
# 词窗大小分别为3,4,5
cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
cnn1 = MaxPooling1D(pool_size=19)(cnn1)
cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
cnn2 = MaxPooling1D(pool_size=18)(cnn2)
cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
cnn3 = MaxPooling1D(pool_size=17)(cnn3)
# 合并三个模型的输出向量
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(3, activation='softmax')(drop)  #一共3个类别
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
one_hot_labels = np_utils.to_categorical(y_train, num_classes=3)  # 将标签转换为one-hot编码
model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10)
#y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
# result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
# result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
# y_predict = list(map(str, result_labels))
# print('准确率', metrics.accuracy_score(y_test, y_predict))
# print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
from keras.utils.vis_utils import plot_model
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=False)



model.save('model_m2_textCNN.h5')