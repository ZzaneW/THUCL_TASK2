# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 01:13:07 2021

@author: T90
"""

#通过训练集和测试机获取其中的语料
import json
import jieba
def get_stopword():
    with open("hit_stopwords.txt", "r", encoding='UTF-8') as f:  # 打开文件
        data = f.read()  # 读取文件
        #print(data)
    return data
def stopword(str_list,stopwords):
    res = ''
    for i in str_list:
        if(i not in stopwords):
            res+=i
            res+=' '
    return res



    

with open('train.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)


stopword_list = get_stopword()
for i in json_data:    
    ss = i['fact']
    str_list = jieba.cut(ss)  #分词
    outstr = stopword(str_list,stopword_list)#停词
    with open('corpus.txt', 'a+', encoding='utf-8') as f:
        f.write(outstr)  # 读取的方式和写入的方式要一致
        f.write('\n')

with open('dev.json','r',encoding='utf8')as fp:
    json_data2 = json.load(fp)
for i in json_data2:    
    ss = i['fact']
    str_list = jieba.cut(ss)  #分词
    outstr = stopword(str_list,stopword_list)#停词
    with open('corpus.txt', 'a+', encoding='utf-8') as f:
        f.write(outstr)  # 读取的方式和写入的方式要一致
        f.write('\n')
