from collections import Counter
import math
import random 
import pandas as pd
'''
Author: LawsonAbs
Date: 2021-09-04 22:07:40
LastEditTime: 2021-09-21 20:57:00
FilePath: /daguan/risk_data_grand/utils/tools.py
'''
import os
from queue import Queue
import re


label2id = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}


bad_clz = ['6-20','5-24','1-1','6-8','5-22','6-13',
            '1-9','6-31','2-11','2-14','6-19','6-28','4-7',  # 13

        # less sample                 
            '3-5','8-27','7-16','8-18','9-23','10-26','2-17' # 7
            ]

bad_clz2id={'6-20': 0, '5-24': 1, '1-1': 2, '6-8': 3, '5-22': 4, '6-13': 5, '1-9': 6, '6-31': 7, '2-11': 8, '2-14': 9, '6-19': 10, '6-28': 11, '4-7': 12, '3-5': 13, '8-27': 14, '7-16': 15, '8-18': 16, '9-23': 17, '10-26': 18, '2-17': 19}

bad_id2clz = {0: '6-20', 1: '5-24', 2: '1-1', 3: '6-8', 4: '5-22', 5: '6-13', 6: '1-9', 7: '6-31', 8: '2-11', 9: '2-14', 10: '6-19', 11: '6-28', 12: '4-7', 13: '3-5', 14: '8-27', 15: '7-16', 16: '8-18', 17: '9-23', 18: '10-26', 19: '2-17'}


def test():
    import pandas as pd
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    pd.set_option('max_colwidth',1000)

    path = "/home/lawson/program/daguan/risk_data_grand/data/train.csv"
    f = pd.read_csv(path,sep=',') 
    # print(f.head)

    labels = f['label']
    text = f['text']
    # print(text)

    cnt = 0
    for text, label in zip(f['text'], labels):
        if cnt < 10:
            print(f"text={text},label = {label}")
            cnt+=1
        else:
            break



def get_vocab_map(vocab_path):
    vocab_path = "/home/lawson/program/daguan/bert-base-fgm/vocab.txt"
    vocab_id = {}
    index = 0
    # 写一个获取vocab映射的
    with open(vocab_path, 'r', encoding='utf-8') as f1:
        for line in f1:            
            line = line.strip("\n")
            vocab_id[line] = index
            index += 1
    
    for item in vocab_id.items():
        key ,value = item
        print(key,value)


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
def ensemble(data_path):    
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
        if cur_file_path == "/home/lawson/program/daguan/res/submission_ck_10_ls_0.01_50000_0.59265.csv" :
            cnt = 1.3
        elif cur_file_path == "/home/lawson/program/daguan/res/submission_0.589.csv":
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
    for _ in matrix:
        print(_)
    res = pd.DataFrame({'id':res_id,
                            'label':res_label})
    submit_path = f"{data_path}/submission_less_ensemble_2.csv" # 合并少数样本的类
    res.to_csv(submit_path,index=False)


# 数据增强方法，用于将小样本的数据扩增
# 将扩增后的数据达到200条
def data_augment():
    less_clz = ['5-24','8-27','6-20','7-16','8-18','9-23','10-26','3-5']
    train_data_path = "/home/lawson/program/daguan/risk_data_grand/data/train.txt"
    train_augment_path = "/home/lawson/program/daguan/risk_data_grand/data/train_augment.txt"
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
        print(len(gene_data))
        gene_data = gene_data[0: 200]
        all_data.extend(gene_data)
    
    all_data.extend(raw_data)
    random.shuffle(all_data)
    with open(train_augment_path,'w') as f:
        idx = 0
        for line in all_data:
            f.write(str(idx)+"\t"+line)
            idx+=1


# 从train.txt中按照类别随机选择数据，使得数据集在类别中达到分布均衡
# # 除了少数标签的类别外，按照rate 的比率取值
def select_data(data_path,rate):
    less_clz_ = ['5-24','8-27','6-20','7-16','8-18','9-23','10-26','3-5'] # 少数目的类别
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
    
    # random.seed(10) # 设置种子，保持随机一致性  => 这里的一致性没有什么用吧
    for item in clz_data.items():
        label,cont = item # 得到label, cont
        # 使用label 和 cont
        random.shuffle(cont)
        print(f"当前正在处理的label文件是:{label}")
        for i in range(int(len(cont)*rate)):
            select_cont.append(cont[i]) # 将其放入到选择结果中

    random.shuffle(select_cont) # shuffle一下
    rate = str(rate) 
    # 得到一个均衡的样本集合，并写入文件
    with open(f'/home/lawson/program/daguan/risk_data_grand/data/train_balance_{rate}_repeat.txt','w') as f:
        for row in select_cont:
            f.write(row)

# 从train.txt中取出某些类别的数据，这么做是因为想在差的类别上再做分类
def select_bad_sample(data_path):
    bad_sample = [] # 
    with open(data_path, 'r') as f:
        for row in f:
            temp = row.strip("\n").split("\t")
            label = id2label[int(temp[2])] 
            # print(row)
            if label in bad_clz: 
                bad_sample.append(temp[0]+"\t"+temp[1]+"\t"+str(bad_clz2id[label])+'\n') # 读入数据        

    random.shuffle(bad_sample)    
    with open(f'/home/lawson/program/daguan/risk_data_grand/data/bad_sample.txt','w') as f:
        for row in bad_sample:
            f.write(row)



# 合并submission，将submission_balance.txt 和 submission_ensemble.txt 合并成一个文件。合并原则是：如果 submission_balance.txt 中预测的样本是少样本，则采取该类别，否则使用submission_ensemble.txt 中的内容
def combine_submission(best_path,path_balance):
    less_clz = ['3-5','8-27','6-20','7-16','8-18','9-23','10-26','5-24'] # 少数目的类别    
    ensemble = {}    
    with open(best_path,'r' ) as f:
        f.readline()
        for line in f:
            iid,label = line.strip("\n").split(",")
            ensemble[iid] = label

    with open(path_balance,'r' ) as f:
        f.readline()
        for line in f:
            iid,label = line.strip("\n").split(",")
            if label in less_clz: 
                ensemble[iid] = label
                
        
    iid = list(ensemble.keys())
    labels = list(ensemble.values())
    temp = pd.DataFrame({'id':iid,
                            'label':labels})
    submit_path = "/home/lawson/program/daguan/submission_best_combine.csv"
    temp.to_csv(submit_path,index=False)
        

# 将预测坏的结果剔除掉，使用专用模型来解决这个问题
# two-step 来做这件事
def combine_bad_sample(submission_bset,submission_bad):
    best = {}
    with open(submission_bset,'r' ) as f:
        f.readline()
        for line in f:
            iid,label = line.strip("\n").split(",")
            best[iid] = label

    # 将坏的预测结果替换掉
    with open(submission_bad,'r' ) as f:
        f.readline()
        for line in f:
            iid,label = line.strip("\n").split(",")
            best[iid] = label
    
    iid = list(best.keys())
    labels = list(best.values())

    temp = pd.DataFrame({'id':iid,
                            'label':labels})
    submit_path = "/home/lawson/program/daguan/submission_two_step.csv"
    temp.to_csv(submit_path,index=False)


# 找出test中的剩余行数
def get_remain_line_in_test(test_path,submission_path):
    test = []
    with open(test_path, 'r') as f:
        for line in f:
            test.append(line)
    
    remain_test = '/home/lawson/program/daguan/risk_data_grand/data/reamin_test.txt'
    remain_iid = []
    with open(submission_path, 'r') as f:
        f.readline()
        for line in f:
            iid,label =line.strip("\n").split(',') #分割
            iid = int(iid)
            if label in bad_clz: # 如果预测结果在 bad_label 中
                remain_iid.append(iid)
    
    with open(remain_test,'w') as f:
        for i in remain_iid:
            line = test[i]
            f.write(line)




'''
针对样本不均衡的类别后处理该怎么写？
01.找出关键字，关键词。如果出现在train中的比例有，且同样出现在test中，那么就使用该关键字作为判别方法
例如我们找出某一类别的关键字，然后判断其出现在train中的比例。找出该类别下的共性关键字，然后来判断。
'''
def post_process():
    pass


def show_res(submission_path):
    label_cnt = {} # label => cnt
    all_label = []
    with open(submission_path,'r') as f:
        f.readline()
        for line in f:
            iid,label = line.strip().split(",")
            # print(label)
            all_label.append(label)
    label_cnt = Counter(all_label)
    a = sorted(label_cnt.items(),key=lambda x:x[1],reverse=True) 
    for item in a:
        print(item)
        # print(label_cnt)


if __name__ == '__main__':
    submission_path = "/home/lawson/program/daguan/res/submission_60000_10.csv"
    # show_res(submission_path)
    # get_vocab_map("")
    # ensemble("/home/lawson/program/daguan/less")
    train_data_path = '/home/lawson/program/daguan/risk_data_grand/data/train.txt'
    test_data_path = '/home/lawson/program/daguan/risk_data_grand/data/test.txt'
    ensemble_path = "/home/lawson/program/daguan/res/submission_ensemble_final.csv"
    less_clz_path = "/home/lawson/program/daguan/less/submission_less_ensemble.csv"
    # select_data(train_data_path,rate=0.36)
    combine_submission(ensemble_path, less_clz_path )
    data_path = '/home/lawson/program/daguan/risk_data_grand/data'
    # get_files_in_class(data_path)
    # select_bad_sample(train_data_path)
    # get_remain_line_in_test(test_path=test_data_path,submission_path= "/home/lawson/program/daguan/submission_ck_10_ls_0.01_50000_0.59265.csv")
    submission_best_path = "/home/lawson/program/daguan/submission_ck_10_ls_0.01_50000_0.59265.csv"
    submission_bad_path = "/home/lawson/program/daguan/submission_bad_sample_1.csv"
    # combine_bad_sample(submission_best_path, submission_bad_path)
    # data_augment()