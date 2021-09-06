'''
Author: LawsonAbs
Date: 2021-09-06 09:14:54
LastEditTime: 2021-09-06 09:14:55
FilePath: /daguan/ml/tf_idf.py
'''

import re
import json
'''
使用TF-IDF挑选出100个最有特征的词条，然后使用逻辑回归进行分类
01.path 是大文本，里面包含多篇文档
02.
'''
def read_doc(path):
    DF = {} # DF{t}记录有多少篇文档包含单词t
    TF = [] # TF{t,d} 记录所有的文档中单词出现的频次信息
    word=() # 记录所有的word 
        
    # 先计算出DF 的信息
    with open(path,'r',encoding='utf-8') as f:
        for line in f: 
            cur_TF = {}  # 记录当前doc中，单词t出现的频次
            cur_words = [] # 记录当前这篇文档所有的单词
#             print(line)
            # 得到本doc 的所有文本
            line = json.loads(line) #加载成dict
            title = line['title']
            content = line['content']
            words = title + " 。 " + content
            words = re.split('[，。 ！？]',words) # 得到本篇文档中所有的词
            for word in words :
                if word !=" " and word !="":
                    if word not in cur_TF.keys():
                        cur_TF[word] = 1
                    else:
                        cur_TF[word] += 1
                    cur_words.append(word)
            TF.append(cur_TF) # 将信息放到TF中
            
            # 更新DF的值
            for word in cur_words:
                if word not in DF.keys():
                    DF[word] = 1
                else:
                    DF[word] +=1
    res = [] # 存储各个doc下的词特征
#     print("DF=",DF)
    # 遍历所有的doc，依次找出top 100的TF-IDF 单词
    for cur_TF in TF:
#         print(cur_TF)
        cur_val = {}
        for item in cur_TF.items():
            word,freq = item
#             print(word,freq)
            if word in DF.keys():                
                val = freq / DF[word]
                cur_val[word] = val # 
#                 print("val",val)
        cur_res = sorted(cur_val.items(),key=lambda x:x[1],reverse=True)
        cur_words = []
        top_k = 3 # 找出前100个词
        for index in range(0,top_k):
#             print(cur_res[index])
            word,key = cur_res[index]
            cur_words.append(word)
            
        res.append(cur_words)
    print(res)


if __name__ == '__main__':
    read_doc("open.txt")