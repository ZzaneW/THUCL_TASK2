# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:45:59 2021

@author: T90
"""
import numpy as np
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re

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
    
def personal_attr_to_tag(str):  #讲类型转化数值型的标签
    if(str=='自然人'):
        return 0
    elif(str=='法人'):
        return 1
    elif(str=='其他组织'):
        return 2
    else:
        return 3  #错误
    
def get_category(num):
    if(num==0):
        return ['自然人']
    elif(num==1):
        return ['法人']
    elif(num==2):
        return ['其他组织']


def get_stopword():
    with open("Method2_textCNN/hit_stopwords.txt", "r", encoding='UTF-8') as f: # 打开文件
    #with open("hit_stopwords.txt", "r", encoding='UTF-8') as f:
        data = f.read()  # 读取文件
        #print(data)
    return data

def stopword(str_list,stopwords):
    res = []
    for i in str_list:
        if(i not in stopwords):
            res.append(i)
    return res    


def predict_sentence(ss):
    vocab = np.load('Method2_textCNN/vocab.npy',allow_pickle=True).item()
    #vocab = np.load('vocab.npy',allow_pickle=True).item()
    str_list = jieba.cut(ss)
    words = stopword(str_list,get_stopword())
    words_id = []
    for i in words:
        if(i in vocab):
            words_id.append(vocab[i])
        else:
            words_id.append(0)
    words_padded=pad_sequences([words_id,words_id],maxlen=20)
    model=load_model('Method2_textCNN/model_m2_textCNN.h5')
    #model=load_model('model_m2_textCNN.h5')
    result = model.predict(words_padded)
    result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
    y_predict = list(map(str, result_labels))
    return get_category(int(y_predict[0]))


def get_sentence(ss):  #获得描述原告与被告基本属性的句子
    infos = ss.split('\n')[0]  #句子都存在于第一段中
    #先将信息中的空格都替换成逗号，方便正则的表达
    infos = infos.replace(' ','，')
    infos = infos.replace(',','，') #逗号统一替换为中文
    #如果infos最后结尾没有标点符号将其补齐
    if(infos[-1]!='。' and infos[-1]!='，'):
        infos+='。'
    if(infos[-1]=='，'):
        infos = infos[:-1]+'。'
    pattern = re.compile(r'(原告.*?。)')   
    result_yuangao = re.findall(pattern,infos)
    pattern2 = re.compile(r'(被告.*?。)')   
    result_beigao = re.findall(pattern2,infos)
    
#    #替换策略1：原告正常匹配，被告由于最后结尾没有句号匹配不上，就去匹配逗号
#    if(len(result_yuangao)>0 and len(result_beigao)==0):
#        pattern3 = re.compile(r'(被告.*，)')
#        result_beigao = re.findall(pattern3,infos)
#    #替换策略2：信息里全部都是逗号没有句号，根据逗号匹配
#    elif(len(result_yuangao)==0 and len(result_beigao)==0):
#        pattern4 = re.compile(r'(原告.*)被告')
#        pattern5 = re.compile(r'(被告.*，)')
#        result_yuangao = re.findall(pattern4,infos)
#        result_beigao = re.findall(pattern5,infos)
    return result_yuangao,result_beigao



def predict(ss):
    result_yuangao,result_beigao = get_sentence(ss)
    labels_yuangao = []
    labels_beigao = []
    for i in result_yuangao:
        labels_yuangao.append(predict_sentence(i)[0])
    labels_yuangao = list(set(labels_yuangao))
    for i in result_beigao:
        labels_beigao.append(predict_sentence(i)[0])
    labels_beigao = list(set(labels_beigao))
    return labels_yuangao,labels_beigao

   

import json
import pandas as pd
from sklearn import metrics
def get_accuracy_and_f1():
    print('计算中--------')
    with open('dev.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
    df_test = pd.DataFrame()
    for i in json_data:
        ss = i['fact']
        pred_chujie,pred_jiekuan = predict(ss)
        d = {
          'label_jiekuan':get_label(i['attr']['借款人基本属性']),
          'label_chujie':get_label(i['attr']['出借人基本属性']),
          'sentence':i['fact'],
          'pred_jiekuan':get_label(pred_jiekuan),
          'pred_chujie':get_label(pred_chujie)
              }
        df_test = df_test.append(d,ignore_index=True)
    
    df_test['sentence'] = df_test['sentence'].astype(str)
    df_test['label_jiekuan'] = df_test['label_jiekuan'].astype(int)
    df_test['label_jiekuan'] = df_test['label_jiekuan'].astype(str)
    df_test['label_chujie'] = df_test['label_chujie'].astype(int)
    df_test['label_chujie'] = df_test['label_chujie'].astype(str)
    df_test['pred_jiekuan'] = df_test['pred_jiekuan'].astype(int)
    df_test['pred_jiekuan'] = df_test['pred_jiekuan'].astype(str)
    df_test['pred_chujie'] = df_test['pred_chujie'].astype(int)
    df_test['pred_chujie'] = df_test['pred_chujie'].astype(str)


    print('借款人基本属性准确率', metrics.accuracy_score(df_test['label_jiekuan'], df_test['pred_jiekuan']))
    print('借款人基本属性平均f1-score:', metrics.f1_score(df_test['label_jiekuan'], df_test['pred_jiekuan'], average='weighted'))


    print('出借人基本属性准确率', metrics.accuracy_score(df_test['label_chujie'], df_test['pred_chujie']))
    print('出借人基本属性平均f1-score:', metrics.f1_score(df_test['label_chujie'], df_test['pred_chujie'], average='weighted'))

#get_accuracy_and_f1()



    

