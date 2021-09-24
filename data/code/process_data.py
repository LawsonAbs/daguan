import math
import random
import re
import os
import torch as t
import torch.nn as nn
from collections import Counter
from math import log
from collections import Counter 
from tqdm import tqdm
import json
import time
import pandas as pd

label2id = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}


label = ['5-24', '6-34', '1-1', '6-8', '10-26', '2-3', '5-22', '6-28', '8-18', '1-4', '2-6', '6-21', '7-16', '6-29', '6-20', 
              '6-15', '6-13', '9-23', '5-35', '2-33', '5-30', '1-9', '8-27', '1-10', '6-19', '3-5', '2-2', '4-7', '2-17', '5-12', 
              '6-32', '6-31', '2-25', '2-11', '2-14']
    

def text_clean(string):
    useless_str = ['\u20e3', '\ufe0f', '\xa0', '\u3fe6', '\U00028482',
                   '\U0002285f', '\ue40d', '\u3eaf', '\u355a', '\U00020086']

    for i in useless_str:
        string = string.replace(i, '')

    return string


def write(sent_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i in sent_list:
            f.write(i+'\n')


def get_all_sent(path, is_test=False):
    sentence = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            sent = json.loads(line)
            text_id = sent['text_id']
            query = sent['query']
            query = text_clean(query)
            query = jio.parse_location(query, town_village=False)['full_location']
            if query == '':
                query = '空'
            candidate = sent['candidate']
            for i in candidate:
                i = json.dumps(i)
                s = json.loads(i)
                text_ = s['text']
                text_ = text_clean(text_)
                text_ = jio.parse_location(text_, town_village=False)['full_location']
                if text_ == '':
                    text_ = '空'
                if is_test:
                    sent_ = query + '\t' + text_
                else:
                    label_ = s['label']
                    if label_ == '不匹配':
                        tgt = '0'
                    elif label_ == '部分匹配':
                        tgt = '1'
                    elif label_ == '完全匹配':
                        tgt = '2'
                    sent_ = query + '\t' + text_ + '\t' + tgt
                sentence.append(sent_)
    return sentence


def get_pretrain_all_csv(path, is_test=False):
    sentence = []
    f = pd.read_csv(path)
    if is_test:
        labels = ['']*len(f['id'])
    else:
        labels = f['label']
    for text, label in zip(f['text'], labels):
        if is_test:
            sent_ = text
        else:
            sent_ = text + '\t' + str(label2id[label])
        sentence.append(sent_)
    return sentence



def get_all_csv(path, is_test=False):
    sentence = []
    f = pd.read_csv(path)
    if is_test:
        labels = ['']*len(f['id'])
    else:
        labels = f['label']
    for id, text, label in zip(f['id'],f['text'], labels):
        if is_test:
            sent_ = str(id) + '\t' + text
        else:
            sent_ = str(id) + '\t' + text + '\t' + str(label2id[label])
        sentence.append(sent_)
    return sentence


def convert_data_list():
    print('>> convert data')
    start_time = time.time()
    # train_path = '../Xeon3NLP_round1_train_20210524.txt'
    # train2_path = '../round2_train.txt'
    train_path = '/tcdata/round1_train.txt'
    train2_path = '/tcdata/round2_train.txt'
    test_path = '/tcdata/test2_release.txt'

    out_train_path = '/data1/liushu/ccks2021_track3_baseline-main/data/train.txt'
    out_train_path2 = '/data1/liushu/ccks2021_track3_baseline-main/data/train2.txt'
    out_test_path = '/data1/liushu/ccks2021_track3_baseline-main/data/test2.txt'

    train_sentence = get_all_sent(train_path)
    test_sentence = get_all_sent(test_path, True)
    train2_sentence = get_all_sent(train2_path)
    
    write(train_sentence, out_train_path)
    write(train2_sentence, out_train_path2)
    write(test_sentence, out_test_path)
    print(f'cost time={time.time()}-{start_time} s')


# 合并train.txt 和 test.txt 成一个纯内容文件，即去掉其中的id，和train里面的标签
def concat_data(train_path,test_path,out_path ):
    print("合并train.txt + test.txt 成all.txt")
    all_content = [] # 记录train + test 中的数据集
    with open(train_path, 'r') as f_train:
        for line in f_train:
            line = line.strip("\n").split('\t')[1]
            all_content.append(line)

    with open(test_path, 'r') as f_test:
        for line in f_test:
            line = line.strip().split('\t')[1]
            all_content.append(line)    
    with open(out_path,'w') as f:
        for line in all_content:
            f.write(line+"\n")
    

def convert_data():
    print('>> convert data')
    start_time = time.time()
    train_path = '../raw_data/datagrand_2021_train.csv'
    test_path = '../raw_data/datagrand_2021_test.csv'
        
    train_sentence = get_pretrain_all_csv(train_path)
    test_sentence = get_pretrain_all_csv(test_path, True)

    # 生成训练的数据
    out_train_path = '../user_data/data/train.txt'
    out_test_path = '../user_data/data/test.txt'

    train_sentence = get_all_csv(train_path)
    test_sentence = get_all_csv(test_path, True)

    write(train_sentence, out_train_path)
    write(test_sentence, out_test_path)
    
    print(f'数据已转换完成，花费时间={time.time()-start_time} s')


# 将无标签数据分割成均匀小份，因为显卡等资源限制，使用第一份数据用于初始的预训练
def split_data():
    print(">>处理无标签数据，获取其前50w行作为初始预训练数据集。")
    source = "../raw_data/datagrand_2021_unlabeled_data.json"
    temp = []
    cnt = 0
    with open(source,'r',errors='ignore') as f:
        line = f.readline()
        while line:
            line = json.loads(line,strict=False)
            # print(line)
            title = line['title']
            content = line['content']
            all = title+"。"+content +"\n"
            temp.append(all)
            cnt +=1
            if (cnt % 500000 == 0 ):
                dest = "../user_data/data/" + str(cnt//500000) + ".txt"
                with open(dest,'a',encoding='utf-8') as file:
                    for raw in temp:
                        file.write(raw)
                    return # 写完第一份数据，直接退出
            #         temp  = []
            
            line = f.readline()
    print(">>初始预训练数据集 0.txt 分割成功！")


# 找出文本中的所有标点字符
def readTxt(path):
    punc = set()
    with open(path,'r',errors='ignore') as f:
        cnt = 0
        for line in tqdm(f,total=14939045):
            # print(line)
            # try :
            line = json.loads(line,strict=False)            
            # except:
            #     print(line)
            #     exit
            title = line['title']
            content = line['content']
            cont = title+" 。 "+content +"\n"            
            # cont = cont.replace('。',' ')
            li = cont.split() # 以空格分
            for word in li:
                try:
                    if int(word):
                        pass
                except:
                    punc.add(word)            
            if cnt > 500*100:
                break
            cnt+=1
    print(punc)


# 找出文本中所有的数字
def get_all_num(path):
    nums = set()    
    with open(path,'r',) as f:
        for line in tqdm(f,total=14939045):
            line = json.loads(line)
            title = line['title']
            content = line['content']
            
            cont = title+" 。 "+content +"\n"
            cont = cont.replace("，"," ").replace("。"," ").replace("！"," ").replace("？"," ")
            
            li = cont.split() # 以空格分
            for word in li:
                try:
                    num = int(word)
                    if num:
                        nums.add(num)
                except:
                    continue            
            if len(nums) > 21000 :
                break
    out_path = "vocab.txt"
    with open(out_path,'w') as f:
        for num in nums:
            f.write(str(num)+"\n")



# 分析提交结果
def analysis_submission(path):
    cls_id = {}
    with open(path ,'r') as  f:
        f.readline()
        for line in f:
            line = line.strip("\n")
            line = line.split(",")
            cls = line[-1]
            if cls not in cls_id.keys():
                cls_id[cls] = 1
            else:
                cls_id[cls] += 1
    sorted(cls_id.items(),key = lambda x : x[1],reverse=True)
    for item in cls_id.items():
        key,value = item
        print(key,value,value/6005)



# 从train.txt中按照类别随机选择数据，使得数据集在类别中达到分布均衡
# # 除了少数标签的类别外，按照rate 的比率取值
def select_data(data_path,rate):    
    less_clz_id = ['0','22','14','12','8','17','4','25']
    select_cont = [] # 被选中的内容
    clz_data = {} # 每个类对应的数据

    with open(data_path, 'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")            
            label = temp[2]
            if label in less_clz_id: # 找出少样本的数据，直接放入
                for i in range(10): 
                    select_cont.append(row) # 读入数据

            else:
                if label not in clz_data.keys():
                    clz_data[label] = []
                else:
                    clz_data[label].append(row) # 读入数据

    for item in clz_data.items():
        label,cont = item # 得到label, cont
        # 使用label 和 cont
        random.shuffle(cont)
        # print(f"当前正在处理的label文件是:{label}")
        for i in range(int(len(cont)*rate)):
            select_cont.append(cont[i]) # 将其放入到选择结果中

    random.shuffle(select_cont) # shuffle一下
    rate = str(rate) 
    # 得到一个均衡的样本集合，并写入文件
    with open(f'../user_data/data/train_balance_{rate}_repeat.txt','w') as f:
        for row in select_cont:
            f.write(row)



# 数据增强方法，用于将小样本的数据扩增
# 将扩增后的数据达到200条
def data_augment():
    less_clz = ['5-24','8-27','6-20','7-16','8-18','9-23','10-26','3-5']
    train_data_path = "../user_data/data/train.txt"
    train_augment_path = "../user_data/data/train_augment.txt"
    raw_data = []
    all_data = [] # 保存数据扩增后的内容
    with open(train_data_path,'r') as f:
        for line in f:
            line = line.strip("\n").split("\t")
            cont = line[1] +"\t"+ line[2] + "\n"
            raw_data.append(cont)
        
    for clz in less_clz: # 扩增当前这类的数据
        gene_data = [] # 生成的数据
        clz_data = [] # 记录当前clz 的data
        for line in raw_data:
            temp = line.strip("\n").split("\t")
            label = temp[1]
            content = temp[0] # 该行文本
            if id2label[int(label)]==clz: # 说明是需要扩增的类别，那么就作为种子数据
                clz_data.append(content) # 放入文本
                temp = content.split("，")
                temp = [_.strip() for _ in temp]
                # 随机shuffle 然后拼接得到一句话
                if len(temp) >= 2:
                    cnt = math.factorial(len(temp))
                    cnt = min(10,cnt)
                    while(cnt):
                        random.shuffle(temp)
                        content = "，".join(temp)
                        cnt-=1
                        content = content + "\t"+label+"\n"
                        gene_data.append(content) # 生成的数据
                elif len(temp) == 2:            
                    content = temp[0]+"，"+temp[1]+"\t"+label+"\n"
                    gene_data.append(content) # 生成的数据
        # print(len(gene_data))
        gene_data = gene_data[0: 200]
        all_data.extend(gene_data)
    
    all_data.extend(raw_data)
    random.shuffle(all_data)
    with open(train_augment_path,'w') as f:
        idx = 0
        for line in all_data:
            f.write(str(idx)+"\t"+line)
            idx+=1


# 这个文件用于后面的vocab_1.txt 的生成以及模型的预训练
def get_10w_unlabel2file():
    num = 0
    title = []
    content = []
    with open('../raw_data/datagrand_2021_unlabeled_data.json','r') as f:
        for line in tqdm(f):
            if num > 100000:
                break
            line = json.loads(line)
            # if line['content'] == '':
            #     # print(line)                
            title.append(line['title'])
            content.append(line['content'])
            num += 1

    unlabeled_data = pd.DataFrame({'title':title,
                               'content':content})

    unlabeled_data.to_csv('../user_data/data/datagrand_unlabeled_data_10w.csv', index=False)


if __name__ == '__main__':
    # step1.转换数据
    convert_data()
    # step2.为预训练准备数据
    split_data() # 划分初始预训练的数据集
    train_path = "../user_data/data/train.txt"
    test_path = "../user_data/data/test.txt"
    out_path = "../user_data/data/all.txt" # 合并成最后的文件
    print(">>将train.txt和test.txt合并成一个文件")
    concat_data(train_path, test_path, out_path)  # 合并得到主要的预训练数据集
    get_10w_unlabel2file() # 使用unlabel的数据生成10w条预训练的数据

    # step3.为微调准备数据
    train_path = "../user_data/temp/train.txt"
    rates = [0.34,0.36] # 按照0.34、 0.36 的比例抽取数据    
    for rate in rates:
        print(f">>正以rate={rate}的比例采样数据...")
        select_data(train_path,rate)

    print(">>数据增强。将少标签的数据按照“，”重新排列得到新的训练数据")
    data_augment() # 生成增强数据
    