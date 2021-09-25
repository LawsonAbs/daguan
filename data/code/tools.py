from collections import Counter
import math
import random 
import pandas as pd
'''
Author: LawsonAbs
Date: 2021-09-04 22:07:40
LastEditTime: 2021-09-25 09:47:16
FilePath: /data/code/tools.py
'''
import os
from queue import Queue
import re


label2id = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}





# 读取所有训练数据，并返回其内容和标签
def read_train_data(data_path):        
    all_cont = []
    labels = [] # 序号id
    with open(data_path, 'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")
            line = temp[1]
            label = temp[2]
            cont = re.split(r'[，。！？ ]',line)
            cur_cont = [ word for word in cont if word != '']
            all_cont.append(cur_cont)
            labels.append(int(label))
    return all_cont,labels


# 按照类别将train中的内容写到各个文件夹中
def get_files_in_class(data_path):
    train_data_path = os.path.join(data_path,"train.txt") # 拼凑得到最后的文件
    id_cont = {} # id=> cont list

    with open(train_data_path,'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")
            cur_cont = temp[1] # 得到当前行的内容
            label = temp[2]            
            if label not in id_cont.keys():
                id_cont[label] =[]
            id_cont[label].append(cur_cont)
    
    # 遍历当前所有的类别，然后生成文件夹+文件
    for item in id_cont.items():
        label,cont_list = item                
        label = int(label)
        idx = 0
        cur_file_dir = os.path.join(data_path,id2label[label]) # 生成类别信息文件夹
        # if os.path.exists(cur_file_dir):
        #     os.removedirs(cur_file_dir) # 删除dir TODO 需要强制删除dir
        if not os.path.exists(cur_file_dir):
            os.makedirs(cur_file_dir)
        
        for cont in cont_list: # 按照内容，逐行写入
            cur_file_path = os.path.join(cur_file_dir,str(idx)+".txt")
            with open(cur_file_path,'w') as f:
                f.write(cont)
                idx+=1
    

# 读取所有测试数据
def read_test_data(data_path):
    all_cont = []
    with open(data_path, 'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")
            line = temp[1]
            cont = re.split(r'[，。！？ ]',line)
            cur_cont = [ word for word in cont if word != '']
            all_cont.append(cur_cont)
    return all_cont

# 将指定文件下的 submission 任务ensemble
def ensemble(data_path,less):
    # 6004 * 35 的二维矩阵
    matrix = [[0 for j in range(35)] for i in range(6004)]
    files = os.listdir(data_path) # 找出当前这个文件夹下的所有文件
    print(files)
    if len(files) < 2:
        return 
    for file in files:
        cur_file_path = os.path.join(data_path,file)
        if os.path.isdir(cur_file_path):
            continue
        cnt = 1 # 设置权重
        if cur_file_path == "../prediction_result/normal/A_epoch_10_0.592.csv" :
            cnt = 1.3
        elif cur_file_path == "../prediction_result/normal/C_epoch_10_0.589.csv" :
            cnt = 1.2 
        with open(cur_file_path,'r') as f:
            f.readline()
            for line in f:
                line = line.strip("\n").split(",")
                iid, label = line
                iid = int(iid)
                label_id = label2id[label]
                matrix[iid][label_id] += cnt
    
    res_id = []
    res_label = []
    # 读完所有的submission
    for i in range(6004):
        cur_row = matrix[i] # 获取该行值
        idx = cur_row.index(max(cur_row)) # 找到最大值的下标
        res_id.append(i)
        res_label.append(id2label[idx])
    # for _ in matrix:
    #     print(_)
    res = pd.DataFrame({'id':res_id,
                            'label':res_label})
    
    # 判断为哪一类数据的ensemble
    if less:
        submit_path = f"{data_path}/submission_less_ensemble.csv" # 合并少数样本的类
    else:
        submit_path = f"{data_path}/submission_normal_ensemble.csv" # 合并所有样本的类
    res.to_csv(submit_path,index=False)



# 合并submission，将submission_balance.txt 和 submission_ensemble.txt 合并成一个文件。合并原则是：如果 submission_balance.txt 中预测的样本是少样本，则采取该类别，否则使用submission_ensemble.txt 中的内容
def combine_submission(best_path,balance_path):
    less_clz = ['3-5','8-27','6-20','7-16','8-18','9-23','10-26','5-24'] # 少数目的类别    
    ensemble = {}    
    with open(best_path,'r' ) as f:
        f.readline()
        for line in f:
            iid,label = line.strip("\n").split(",")
            ensemble[iid] = label

    with open(balance_path,'r' ) as f:
        f.readline()
        for line in f:
            iid,label = line.strip("\n").split(",")
            if label in less_clz: 
                ensemble[iid] = label
                
        
    iid = list(ensemble.keys())
    labels = list(ensemble.values())
    temp = pd.DataFrame({'id':iid,
                            'label':labels})
    submit_path = "../prediction_result/result.txt"
    temp.to_csv(submit_path,index=False)
    


if __name__ == '__main__':
    
    # show_res(submission_path)
    # get_vocab_map("")
    ensemble("../prediction_result/less",less=True)
    ensemble("../prediction_result/normal",less=False)
    
    ensemble_normal_path = "../prediction_result/normal/submission_normal_ensemble.csv"
    ensemble_less_path = "../prediction_result/less/submission_less_ensemble.csv"
    combine_submission(ensemble_normal_path, ensemble_less_path)