'''
Author: LawsonAbs
Date: 2021-09-06 09:14:54
LastEditTime: 2021-09-06 09:14:55
FilePath: /daguan/ml/tf_idf.py
'''
from visdom import Visdom
from torch.optim import optimizer
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch as t
import torch.nn as nn
import re
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
'''
使用TF-IDF挑选出100个最有特征的词条，然后使用逻辑回归进行分类
01.path 是大文本，里面包含多篇文档
02.tf-idf 
'''
def read_doc(path,top_k):
    DF = {} # DF{t}记录有多少篇文档包含单词t
    TF = [] # TF{t,d} 记录所有的文档中单词出现的频次信息
    word=() # 记录所有的word 
    labels = []
    # 先计算出DF 的信息
    with open(path,'r',encoding='utf-8') as f:
        for line in tqdm(f,total=1000): 
            cur_TF = {}  # 记录当前doc中，单词t出现的频次
            cur_words = [] # 记录当前这篇文档所有的单词
#             print(line)
            # 得到本doc 的所有文本
            line = line.split("\t") # 以\t分割
            words = re.split(r'[，。！？ ]',line[1])
            label = line[2].strip("\n")
            labels.append(int(label))
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
    features = [] # 存储各个doc下的词特征
    # 遍历所有的doc，依次找出top 100的TF-IDF 单词
    for cur_TF in TF:
#         print(cur_TF)
        cur_val = {}
        for item in cur_TF.items():
            word,freq = item
#           print(word,freq)
            val = freq / DF[word]
            cur_val[word] = val # 
#                 print("val",val)
        cur_res = sorted(cur_val.items(),key=lambda x:x[1],reverse=True)
        cur_words = []
        # 选择top_k 和 len(cur_words)中的较小值
        for index in range(0,min(top_k,len(cur_res))):
#             print(cur_res[index])
            word,key = cur_res[index]
            cur_words.append(int(word))
        # 如果长度小于top_k
        while (len(cur_words) < top_k):
            cur_words.append(0) # 插入0字符
        features.append(cur_words)
    # print(res)
    # for item in zip(features,labels):
    #     print(item)
    return features,labels # 返回，并将其作为向量存储

class LR(nn.Module):
    def __init__(self,in_1,out_1,in_2,out_2,in_3,out_3):
        super().__init__()
        self.sigmoid = nn.Sigmoid() # 激活函数
        self.linear_1 = nn.Linear(in_1,out_1)        
        self.linear_2 = nn.Linear(in_2,out_2)
        self.linear_3 = nn.Linear(in_3,out_3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        out = self.linear_1(x) # 执行线性变化
        # out = self.sigmoid(out)        
        out = self.linear_2(out)
        # out = self.sigmoid(out)
        out = self.linear_3(out)        
        logits = self.softmax(out)

        return logits    # shape = 16,35

class FeatureDataset(Dataset):

    def __init__(self,x,label):        
        super().__init__()
        self.x = x
        self.label = label

    def __len__(self):
        return len(self.x)
    
    # 返回
    def __getitem__(self,index):
        return self.x[index],self.label[index]


def train():
    data_path = "/home/lawson/program/daguan/risk_data_grand/data/train.txt"
    x,labels=read_doc(data_path,top_k=64) # 找出重要性前100的词
    
    # 执行sigmoid 将所有的id映射到 (0-1范围)
    # in_fea = []
    # sigmoid = nn.Sigmoid()
    # for i in x:
    #     temp = []
    #     for _ in i:
    #         temp.append(sigmoid(t.tensor(_)))
    #     in_fea.append(temp)

    x = t.tensor(x,dtype=t.float).cuda()
    labels = t.tensor(labels,dtype=t.long).cuda()

    dataset = FeatureDataset(x,labels)
    dataloader = DataLoader(dataset,
                            batch_size=16,
                            )
    model = LR(64,128,128,128,128,35)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

    model.to(device)
    
    optimizer = t.optim.Adam(model.parameters(),lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    num_epoch = 10
    viz = Visdom()
    global_step = 0
    for epoch in range(num_epoch):
        for batch in dataloader:
            x,label = batch
            # 将数据x做归一化处理，如果单纯的使用id，很难训练

            logits = model(x)
            loss = criterion(logits,label)
            viz.line([loss.item()],[global_step],win="loss",update="append")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            for parameters in model.named_parameters():
                name,value = parameters    
                if("linear_1.weight") in  name:
                    print(value[0,0:10])

if __name__ == '__main__':
    train()