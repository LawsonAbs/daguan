from queue import Queue # 使用队列保持top_k
import pandas as pd
import torch as t
from transformers import BertTokenizer,BertModel 
model_path = ""
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

# 使用knn对样本进行分类
def knn(train_path,test_path,k):
    # 读取train
    train_embedding = {} # id -> embedding 
    test_embedding = {}
    id_clz = {} # id=>class 的mapping，记录train的数据
    res = {} # 最后的结果
    submit_path = f"submission_knn_{k}.csv"
    q = Queue(maxsize=k)  # 队列最大的容量为5

    with open(train_path,'r') as f_train:
        for line in f_train: # 读取train
            line = line.strip("\n").split("\t")
            line_id,content,clz = line 
            inputs = tokenizer(content) 
            cur_cont_emb = model(**inputs) #处理输入，得到embedding
            train_embedding[int(line_id)] = cur_cont_emb # 得到对应的id
            id_clz[line_id] = clz

    test_ids  = []
    test_cls = []
    with open(test_path,'r') as f_test:
        for line in f_test: # 读取test
            line = line.strip("\n").split("\t")
            line_id,content,clz = line 
            inputs = tokenizer(content) 
            cur_cont_emb = model(**inputs) #处理输入，得到embedding
            test_embedding[int(line_id)] = cur_cont_emb # 得到对应的id

            # 计算相似度
            max_simi = 0 # 最大的相似度
            for item in train_embedding.items():
                train_id,train_emb = item
                simi = t.cosine_similarity(train_emb,cur_cont_emb,dim = 0) 
                if simi > max_simi:
                    q.put(train_id) # 
                    cur_clz = id_clz[train_id] # 得到类别信息
                    max_simi = simi # 更新simi值

            # 得到最后的结果
            # res[line_id] = cur_clz
            test_ids.append(line_id)
            test_cls.append(cur_clz)
    
    res = pd.DataFrame({'id':test_ids,
                        'label':test_cls})
    res.to_csv(submit_path,index=False)