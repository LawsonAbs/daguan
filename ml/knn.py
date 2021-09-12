'''
Author: LawsonAbs
Date: 2021-09-10 23:02:51
LastEditTime: 2021-09-11 22:50:45
FilePath: /daguan/ml/knn.py
'''
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import os
os.environ['CUDA_VISIBLE_device_gpuS'] = '0'
from queue import Queue # 使用队列保持top_k
import pandas as pd
import torch as t
from transformers import BertTokenizer,BertModel 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
model_path = "/home/lawson/program/daguan/pretrain_model/bert-base-fgm/2.4G+4.8M_large_10000_128_checkpoint-40000"
device_gpu = t.device("cuda:0" if t.cuda.is_available() else "cpu")
device_cpu = t.device("cpu")

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}


# 读取所有训练数据，并返回其中
def read_train_data(data_path):        
    all_cont = []
    labels = [] # 序号id
    with open(data_path, 'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")
            line = temp[1]
            label = temp[2]
            all_cont.append(line)
            labels.append(int(label))
    return all_cont,labels


# 读取所有测试数据
def read_test_data(data_path):
    all_cont = []
    with open(data_path, 'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")
            line = temp[1]     
            all_cont.append(line)
    return all_cont

class MyDataset(Dataset):
    def __init__(self,x,label=None):
        super().__init__()
        self.x = x
        self.label =label
    
    def __getitem__(self, index):
        return self.x[index],self.label[index]
    
    def __len__(self):
        return len(self.x)


# 使用knn对样本进行分类
def knn(train_data,train_labels,dev_data,dev_labels,test_data,k):
    # 读取train
    dev_embedding = {}    
    train_embedding = {} # id -> embedding 
    test_embedding = {}    
    res = {} # 最后的结果
    
    q = Queue(maxsize=k)  # 队列最大的容量为5
    max_macro_f1 = 0
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.to(device_gpu)
    # step1. 使用预训练好的模型得到整个句子的表示
    line_id = 0
    train_dataset = MyDataset(x=train_data,label=train_labels)
    train_dataloader = DataLoader(train_dataset,batch_size=1)

    for batch in tqdm(train_dataloader): # 读取train
        content ,_ =  batch 
        inputs = tokenizer(content,return_tensors='pt',max_length=128,padding='max_length')
        inputs = inputs.to(device_gpu)
        with t.no_grad():
            out = model(**inputs) #处理输入，得到embedding
        cur_cont_emb = out[0] # 取 last_hidden_states
        cur_cont_emb = cur_cont_emb.to(device_cpu)  # 为了防止爆显存
        train_embedding[line_id] = cur_cont_emb[0,0,:] # 得到对应的id
        line_id  += 1

    # step2.使用dev测试效果 当值为k时的效果
    dev_ids  = []
    dev_cls = []
    idx = 0
    for content in tqdm(dev_data): # 读取test                
        inputs = tokenizer(content,return_tensors='pt',max_length=128,padding='max_length') 
        inputs = inputs.to(device_gpu)
        with t.no_grad():
            out = model(**inputs) #处理输入，得到embedding
        cur_cont_emb = out[0] # 拿到cls 位置的向量
        cur_test_emb =  cur_cont_emb[0,0,:]
        cur_test_emb = cur_test_emb.to(device_cpu)
        dev_embedding[idx] = cur_test_emb # 得到对应的id     

        # 计算相似度
        max_simi = 0 # 最大的相似度
        for item in train_embedding.items():
            train_id,train_emb = item
            simi = t.cosine_similarity(train_emb,cur_test_emb,dim = 0) 
            if simi > max_simi:
                cur_clz = train_labels[train_id] # 得到类别信息
                if q.qsize() == k:
                    q.get() # 从头取一个
                q.put(cur_clz) 
                max_simi = simi # 更新simi值

        cls_cnt = [0 for i in range(35)]
        while(not q.empty()):
            cls = q.get()
            cls_cnt[cls] += 1
        # 从queue中找出数目较多的类别        
        # 这里有个问题就是，如果有多个类个数都是相同的，那么这里就直接取了第一个，但合理吗？
        cur_clz = cls_cnt.index(max(cls_cnt)) 
                
        # res[line_id] = cur_clz
        dev_ids.append(idx)
        dev_cls.append(cur_clz)
        idx+=1

    # step3.使用dev检测一下KNN分类效果
    dev_macro_f1 = f1_score(dev_labels,dev_cls,average='macro')
    
    # step4.如果说，dev上面效果不错，那么就可以尝试对 test 做预测，并写入结果
    print(f"k={k},dev_macro_f1={dev_macro_f1}")
    submit_path = f"submission_knn_epoch={k}_{dev_macro_f1}.csv"
    if dev_macro_f1 > max_macro_f1: 
        max_macro_f1 = dev_macro_f1       
        test_ids  = []
        test_cls = []
        idx = 0
        for content in tqdm(test_data): # 读取test                
            inputs = tokenizer(content,return_tensors='pt',max_length=128,padding='max_length') 
            inputs = inputs.to(device_gpu)
            with t.no_grad():
                out = model(**inputs) #处理输入，得到embedding
            cur_cont_emb = out[0] # 拿到cls 位置的向量
            cur_test_emb =  cur_cont_emb[0,0,:]
            cur_test_emb = cur_test_emb.to(device_cpu)
            test_embedding[idx] = cur_test_emb # 得到对应的id     

            # 计算相似度
            max_simi = 0 # 最大的相似度
            for item in train_embedding.items():
                train_id,train_emb = item
                simi = t.cosine_similarity(train_emb,cur_test_emb,dim = 0) 
                if simi > max_simi:
                    cur_clz = train_labels[train_id] # 得到类别信息
                    if q.qsize() == k:
                        q.get() # 从头取一个
                    q.put(cur_clz) 
                    max_simi = simi # 更新simi值

            cls_cnt = [0 for i in range(35)]
            while(not q.empty()):
                cls = q.get()
                cls_cnt[cls] += 1
            # 从queue中找出数目较多的类别        
            # 这里有个问题就是，如果有多个类个数都是相同的，那么这里就直接取了第一个，但合理吗？
            cur_clz = cls_cnt.index(max(cls_cnt)) 
                    
            # res[line_id] = cur_clz
            test_ids.append(idx)
            test_cls.append(cur_clz)
            idx+=1
        res_cls = [id2label[cls] for cls in test_cls]
        res = pd.DataFrame({'id':test_ids,
                            'label':res_cls})
                            
        res.to_csv(submit_path,index=False)


if __name__ == '__main__':
    train_path = "/home/lawson/program/daguan/risk_data_grand/data/train.txt"
    test_path = "/home/lawson/program/daguan/risk_data_grand/data/test.txt"
    
    train_data,train_labels = read_train_data(train_path)
    test_data = read_test_data(test_path)        
    x_train,x_dev,y_train,y_dev = train_test_split(train_data,train_labels,test_size=0.3,random_state=32,shuffle=True)
    # test_queue()
    for i in range(2,5):
        knn(x_train,y_train,x_dev,y_dev,test_data,i)
    