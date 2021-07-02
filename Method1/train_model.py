# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:18:08 2021

@author: T90
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 18:58:48 2021

@author: T90
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 10:37:41 2021

@author: T90
"""
import json
import pandas as pd
import re
import sys
import os
from sklearn.model_selection import train_test_split
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

   
    
def personal_attr_to_tag(str):  #讲类型转化数值型的标签
    if(str=='自然人'):
        return 0
    elif(str=='法人'):
        return 1
    elif(str=='其他组织'):
        return 2
    else:
        return 3  #错误
    
def get_label(per_list):  #讲属性列表转化
    res = []
    for i in per_list:
        res.append(personal_attr_to_tag(i))   #直接追加就可以，不存在重复问题
    if(res==[0]):
        return 0
    elif(res==[1]):
        return 1
    elif(res==[2]):
        return 2
    elif(res==[0,1] or [1,0]):
        return 3
    # elif(res==[0,2]):  #实验发现下面这些情况不存在
    #     return 4
    # elif(res==[1,2]):
    #     return 5
    # elif(res==[1,2,3]):
    #     return 6


with open('train.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
with open('dev.json','r',encoding='utf8')as fp2:
    json_data2 = json.load(fp2)

df_train = pd.DataFrame()
df_test = pd.DataFrame()  

for i in json_data:
    
    d = {
         'label':get_label(i['attr']['借款人基本属性']),
         'sentence':i['fact']
             }
    df_train = df_train.append(d,ignore_index=True)
    
for i in json_data2:
    
    d = {
         'label':get_label(i['attr']['借款人基本属性']),
         'sentence':i['fact']
             }
    df_test = df_test.append(d,ignore_index=True)


# df.to_csv('data.csv', sep=',', header=True, index=True,encoding='utf_8_sig')

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

df_train['sentence'] = df_train['sentence'].astype(str)
df_train['label'] = df_train['label'].astype(int)
df_train['label'] = df_train['label'].astype(str)

df_test['sentence'] = df_test['sentence'].astype(str)
df_test['label'] = df_test['label'].astype(int)
df_test['label'] = df_test['label'].astype(str)

df = df_train.append(df_test)


# df.to_csv('data.csv', sep=',', header=True, index=True,encoding='utf_8_sig')

stopword_list = get_stopword()
cw = lambda x: list(jieba.cut(x))  #分词
st = lambda x: stopword(x,stopword_list)
#停词


df_train['words'] = df_train['sentence'].apply(cw)
df_train['words'] = df_train['words'].apply(st)
df_test['words'] = df_test['sentence'].apply(cw)
df_test['words'] = df_test['words'].apply(st)
df = df_train.append(df_test)
tokenizer=Tokenizer()  #创建一个Tokenizer对象
#fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
tokenizer.fit_on_texts(df['words'])
vocab=tokenizer.word_index #得到每个词的编号
#存储字典
np.save('vocab.npy', vocab)


#x_train, x_test, y_train, y_test = train_test_split(df['words'], df['label'], test_size=0.1)
x_train = df_train['words']
y_train = df_train['label']
x_test = df_test['words']
y_test = df_test['label']
# 将每个样本中的每个词转换为数字列表，使用每个词的编号进行编号
x_train_word_ids=tokenizer.texts_to_sequences(x_train)
x_test_word_ids = tokenizer.texts_to_sequences(x_test)
#序列模式 每条样本长度不唯一，将每条样本的长度设置一个固定值
x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=50) #将超过固定值的部分截掉，不足的在最前面用0填充
x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=50)



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

main_input = Input(shape=(50,), dtype='float64')
# 词嵌入（使用预训练的词向量）

embedder = Embedding(len(vocab) + 1, 300, input_length=50, weights=[embedding_matrix], trainable=False)
embed = embedder(main_input)
# 词窗大小分别为3,4,5
cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(embed)
cnn1 = MaxPooling1D(pool_size=48)(cnn1)
cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(embed)
cnn2 = MaxPooling1D(pool_size=47)(cnn2)
cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(embed)
cnn3 = MaxPooling1D(pool_size=46)(cnn3)
# 合并三个模型的输出向量
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.2)(flat)
main_output = Dense(4, activation='softmax')(drop)  #一共七个类别
model = Model(inputs=main_input, outputs=main_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
one_hot_labels = np_utils.to_categorical(y_train, num_classes=4)  # 将标签转换为one-hot编码
model.fit(x_train_padded_seqs, one_hot_labels, batch_size=800, epochs=10)
#y_test_onehot = keras.utils.to_categorical(y_test, num_classes=3)  # 将标签转换为one-hot编码
result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
y_predict = list(map(str, result_labels))
print('准确率', metrics.accuracy_score(y_test, y_predict))
print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
from keras.utils.vis_utils import plot_model
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=False)



model.save('model_m1_textCNN.h5')