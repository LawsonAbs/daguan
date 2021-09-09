# coding:utf-8
from tqdm import tqdm
import json
import time
import pandas as pd

label_dict = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

label = ['5-24', '6-34', '1-1', '6-8', '10-26', '2-3', '5-22', '6-28', '8-18', '1-4', '2-6', '6-21', '7-16', '6-29', '6-20', 
              '6-15', '6-13', '9-23', '5-35', '2-33', '5-30', '1-9', '8-27', '1-10', '6-19', '3-5', '2-2', '4-7', '2-17', '5-12', 
              '6-32', '6-31', '2-25', '2-11', '2-14']

id_label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}            

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
            sent_ = text + '\t' + str(label_dict[label])
        sentence.append(sent_)
    return sentence


def get_unlabel_pretrain_all_csv(path):
    sentence = []
    f = pd.read_csv(path)
    for title, content in zip(f['title'], f['content']):
        if title == title and content == content:
            sent_ = title + '。' + content
        elif title == title and content != content:
            sent_ = title
        elif title != title and content == content:
            sent_ = content
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
            sent_ = str(id) + '\t' + text + '\t' + str(label_dict[label])
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
    train_path = 'datagrand_2021_train.csv'
    test_path = 'datagrand_2021_test.csv'
    # 生成预训练的数据
    out_train_path = '/data1/liushu/risk_data_grand/data/pretrain_train.txt'
    out_test_path = '/data1/liushu/risk_data_grand/data/pretrain_test.txt'

    train_sentence = get_pretrain_all_csv(train_path)
    test_sentence = get_pretrain_all_csv(test_path, True)

    write(train_sentence, out_train_path)
    # 生成训练的数据
    out_train_path = '/data1/liushu/risk_data_grand/data/train.txt'
    out_test_path = '/data1/liushu/risk_data_grand/data/test.txt'

    train_sentence = get_all_csv(train_path)
    test_sentence = get_all_csv(test_path, True)

    write(train_sentence, out_train_path)
    write(test_sentence, out_test_path)
    
    print(f'cost time={time.time()-start_time} s')


# 将无标签数据分割成均匀样本
def split_data():
    source = "/home/lawson/program/daguan/risk_data_grand/data/datagrand_2021_unlabeled_data.json"
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
            if (cnt % 500000 == 0 and cnt):
                dest = "/home/lawson/program/daguan/risk_data_grand/data/small_json/" + str(cnt//500000) + ".txt"
                with open(dest,'a',encoding='utf-8') as file:
                    for raw in temp:
                        file.write(raw)
                    temp  = []
            cnt +=1
            line = f.readline()

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


# 查看类别是否均衡， 并将所有的每类数据放到专属文件中
def statisfy(path):
    cls_id = {}
    all = 0
    with open(path,'r') as f:
        for line in f:
            line = line.strip("\n")
            line = line.split("\t")
            num = line[-1] # 最后一个为类别
            if num not in cls_id.keys():
                cls_id[num] = 1
            else:
                cls_id[num] +=1
            all+=1
    cls_id = sorted(cls_id.items(),key= lambda x: x[1],reverse=True) 
    print("all=",all)
    for item in cls_id:
        key,value = item
        print(id_label[int(key)],value,value/all,value/all*6005)

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


if __name__ == '__main__':
    # convert_data()
    # split_data()
    # readTxt("/home/lawson/program/daguan/risk_data_grand/data/datagrand_2021_unlabeled_data.json")
    # get_all_num("/home/lawson/program/daguan/risk_data_grand/data/datagrand_2021_unlabeled_data.json")
    # statisfy("/home/lawson/program/daguan/risk_data_grand/data/train.txt")
    # 将
    # index = [i for i in range(35)]
    # id_label = dict(zip(index,id_label))
    # print(id_label)
    # analysis_submission("submission_0906.csv")
    train_path = "/home/lawson/program/daguan/risk_data_grand/data/train.txt"
    test_path = "/home/lawson/program/daguan/risk_data_grand/data/test.txt"
    out_path = "/home/lawson/program/daguan/risk_data_grand/data/all.txt"
    concat_data(train_path, test_path, out_path)