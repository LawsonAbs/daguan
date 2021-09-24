import json
from tqdm import tqdm
'''
Author: LawsonAbs
Date: 2021-09-22 21:56:22
LastEditTime: 2021-09-23 22:25:51
FilePath: /daguan_gitee/code/build_vocab.py
'''
from collections import Counter
import os
import json
from tqdm import tqdm
from collections import Counter
import pandas as pd

def loadData(path):
    allData=[]
    f = pd.read_csv(path)
    for line in f['text']:
        a = []
        a = [int(a) for a in line.split() if a not in ['。', '，', '！', '？']]
        allData.append(a)
    return allData

def loadCsvData(path):
    allData=[]
    f = pd.read_csv(path)
    for title, content in zip(f['title'], f['content']):
        if title == title and content == content:
            a = [int(a) for a in title.split() if a not in ['。', '，', '！', '？']] + \
                [int(a) for a in content.split() if a not in ['。', '，', '！', '？']]
        elif title == title and content != content:
            # print('content=',content)
            a = [int(a) for a in title.split() if a not in ['。', '，', '！', '？']]
        elif title != title and content == content:
            # print('title=',title)
            a = [int(a) for a in content.split() if a not in ['。', '，', '！', '？']]
        else:
            a = []
        allData.append(a)
    return allData


'''
第一种生成vocab的方式：
01. 使用10w无标签数据 + train.txt + test.txt 生成得到。
'''
def generate_vocab_1():
    print(">>开始生成词表vocab_1.txt")
    allData=loadData('../raw_data/datagrand_2021_train.csv') + loadCsvData('../user_data/data/datagrand_unlabeled_data_10w.csv')
    test_data = loadData('../raw_data/datagrand_2021_test.csv')

    model_lists = [""]
    ## 词频大于等于5
    counts=5
    token2count=Counter()
    for i in allData+test_data:
        token2count.update(i)

    
    pre=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',]
    tail=[]
    for k,v in token2count.items():
        if v>=counts:
            tail.append(k)
    tail.sort()
    vocab=pre+tail
    print(f"词频：5，词表大小：{len(vocab)}")
    
    with open("../user_data/data/vocab_1.txt", "w", encoding="utf-8") as f:
        for i in vocab:
            f.write(str(i)+'\n')
    print(">>vocab_1.txt 生成成功")

   

'''
第二种生成vocab的方式

从无标签找出文本中所有的数字，从而生成字典。但实际上，这是不合理的，因为很难把所有的字词都统计完整，而且过于庞大的
字典对模型训练也不友好。
'''
def generate_vocab_2():
    print(">>开始生成词表vocab_2.txt")
    nums = set()    
    out_path = "../user_data/data/vocab_2.txt"
    path = '../raw_data/datagrand_2021_unlabeled_data.json', # 构建vocab_2.txt
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
    
    with open(out_path,'w') as f:
        for num in nums:
            f.write(str(num)+"\n")
    print(">>vocab_2.txt 生成成功")

def main():    
    generate_vocab_1()
    generate_vocab_2()
    

if __name__ == '__main__':
    main()