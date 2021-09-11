#coding: utf-8
import pandas as pd
import json
import re
import os
import time
import random
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import string

# 从文本中挑选出无用的词
# 使用unlabeled 的数据集，基于大数据分析查找出各个词的出现频率，同普通文本进行对比
punctuation = string.punctuation #得到常用的字符
label2id = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}


# 从加密文本中挑选出无用的词
# 使用unlabeled 的数据集，基于大数据分析查找出各个词的出现频率，然后对应的过滤掉 train.txt 文本中的高频无用词，大概400个，所以这里我取400
def get_invalid_words(data_path):
    invalid_words = []
    word_cnt = {} # word => cnt 的dict
    word_freq = {} # word => freq 的dict    
    total = 0             
    files = os.listdir(data_path) # 得到当前文件夹下所有的文件
    # print(files) 
    
    for file in files:
        cur_file_path = os.path.join(data_path,file)
        print("当前处理的文件是：",cur_file_path)
        with open(cur_file_path,'r',encoding="utf8", errors='ignore') as f:
            for cont in f: #
#                 print(cont) # 查看得到的文本内容
                cont= cont.strip("\n")
                cont = re.split(r'[，。！？、 ]',cont)
#                 print(cont)
                for word in cont:
                    if (word not in punctuation) and word !='':
                        if word not in word_cnt.keys(): # 判断它得是一个单词
                            word_cnt[word] = 1
                        else:
                            word_cnt[word] += 1
#                         print(word,end=" ")
                    total += 1                
                
    # print(word_cnt.keys())
    for item in word_cnt.items():
        word,cnt = item
        freq = cnt/total
        word_freq[word] = freq
    word_freq = sorted(word_freq.items(),key = lambda x:x[1],reverse=True) # 按照频率排序
    word_freq_path = "/home/lawson/program/daguan/risk_data_grand/data/small_json/word_freq.json"
    with open(word_freq_path,'w' ) as f:
        json.dump(word_freq,f)
    return invalid_words

# 读取上面生成的数据文件（word_freq.json），返回一个无效却高频单词列表
def read_invalid_words(data_path,top_k):
    invalid_words = [] # 得到高频无效单词
    with open(data_path,'r') as f:
        cont = f.read()
        cont = json.loads(cont)
        # print(cont)
        for line in cont[0:top_k]:
            word, freq = line
            invalid_words.append(word)
    return invalid_words



# 构建文本中的词典
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r') as fp:
        for line in fp.readlines():
            word = line.strip().decode("utf-8")
            if len(word)>0 and word not in words_set: # 去重
                words_set.add(word)
    return words_set


def TextProcessing(folder_path, dev_size=0.2):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []

    # 类间循环
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        # 类内循环 => 处理某个相同分类的所有文本        
        for file in files:
            with open(os.path.join(new_folder_path, file), 'r') as fp:
               raw = fp.read()
            # print raw   
            # 使用某种方法得到 word_list ，比如用TF_IDF，先去除高频词无用词，然后再获取
            word_list = []
            data_list.append(word_list)
            class_list.append(folder.decode('utf-8'))
                
    ## 划分训练集和测试集
    # train_data_list, dev_data_list, train_class_list, dev_class_list = sklearn.cross_validation.train_dev_split(data_list, class_list, dev_size=dev_size)
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*dev_size)+1
    train_list = data_class_list[index:]
    dev_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    dev_data_list, dev_class_list = zip(*dev_list)

    # 统计词频放入all_words_dict
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if all_words_dict.has_key(word):
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # key函数利用词频进行降序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f:f[1], reverse=True) # 内建函数sorted参数需为list
    all_words_list = list(zip(*all_words_tuple_list)[0])

    return all_words_list, train_data_list, dev_data_list, train_class_list, dev_class_list


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    # 选取特征词
    feature_words = []
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000: # feature_words的维度1000
            break
        # print all_words_list[t]
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1<len(all_words_list[t])<5:
            feature_words.append(all_words_list[t])
            n += 1
    return feature_words


def text_features(train_data_list, feature_words, flag='sklearn'):
    def process(text, feature_words):
        text_words = set(text)
        ## -----------------------------------------------------------------------------------
        if flag == 'nltk':
            ## nltk特征 dict
            features = {word:1 if word in text_words else 0 for word in feature_words}
        elif flag == 'sklearn':
            ## sklearn特征 list
            features = [1 if word in text_words else 0 for word in feature_words]
        else:
            features = []
        ## -----------------------------------------------------------------------------------
        return features
    
    # 找出每个文本的特征
    feature_list = [process(text, feature_words) for text in train_data_list]    
    return feature_list


def text_classifier(train_feature_list, dev_feature_list, train_class_list, dev_class_list, flag='sklearn'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        dev_flist = zip(dev_feature_list, dev_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(dev_feature_list)
        # for dev_feature in dev_feature_list:
        #     print classifier.classify(dev_feature),
        # print ''
        dev_accuracy = nltk.classify.accuracy(classifier, dev_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print classifier.predict(dev_feature_list)
        # for dev_feature in dev_feature_list:
        #     print classifier.predict(dev_feature)[0],
        # print ''                        
        dev_predict = classifier.predict(dev_feature_list)
        dev_macro_f1 = f1_score(dev_class_list,dev_predict,average='macro')

        # dev_accuracy = classifier.score(dev_feature_list, dev_class_list)
    else:
        dev_accuracy = []
    return classifier,dev_macro_f1


def other():
    ## 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list, train_data_list, dev_data_list, train_class_list, dev_class_list = TextProcessing(folder_path, dev_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # flag = 'nltk'
    flag = 'sklearn'
    deleteNs = range(0, 1000, 20)
    dev_accuracy_list = []
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, dev_feature_list = TextFeatures(train_data_list, dev_data_list, feature_words, flag)
        dev_accuracy = TextClassifier(train_feature_list, dev_feature_list, train_class_list, dev_class_list, flag)
        dev_accuracy_list.append(dev_accuracy)
    print(dev_accuracy_list)

    # 结果评价
    plt.figure()
    plt.plot(deleteNs, dev_accuracy_list)
    plt.title('Relationship of deleteNs and dev_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('dev_accuracy')
    plt.savefig('result.png')
    print("end")


# 获取某个类别下的高频词，top_k 为前k个词
def get_high_freq_word_in_class(data_path,top_k,invalid_words):    
    all_conts = [] # 所有的文本
    feature_words = set()  # 所有类的高频词构成的一个feature    
    with open(data_path, 'r') as f:
        for line in f:
            all_conts.append(line)
    
    # 依次遍历35 个类别，然后获取不同类别下的高频词
    for clz in range(35):
        word_cnt = {} # words => cnt
        temp_feature = [] # 针对每一类别的feature
        for line in all_conts:
            line = line.strip("\n").split("\t") # 拿到文本
            cont,cur_clz = line[1::] 
            if int(cur_clz) == clz: # 如果是当前的类别
                cont = re.split(r'[！？，。 ]',cont) # 得到列表
                for word in cont:
                    if word  != "" :
                        if word not in word_cnt.keys():
                            word_cnt[word] = 1
                        else:
                            word_cnt[word] +=1 
        
        # 对该类别下的文件单词排序，得到的是list
        res = sorted(word_cnt.items(),key = lambda x : x[1],reverse=True)        
        idx = 0
        
        # 针对每个类别开始
        for item in res:
            if idx > top_k : # 每个文章中只找top_k 个词
                break
            word,cnt = item # 找到词和词频
            if word not in invalid_words: # 如果这个词不在共有词频里，那么就将其加入到feature_words 中
                feature_words.add(word)
                temp_feature.append(word)
                idx+=1
    # print(len(feature_words)) # 特征长度
    return feature_words


# 读取所有训练数据，并返回其中
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


def predict(classifier):
    classifier

if __name__ == '__main__':
    data_path = "/home/lawson/program/daguan/risk_data_grand/data/small_json/word_freq.json"
    train_data_path = "/home/lawson/program/daguan/risk_data_grand/data/train.txt"
    test_data_path = "/home/lawson/program/daguan/risk_data_grand/data/test.txt"
    max_dev_score = 0

    # 遍历超参取最好的模型
    for invalid_k in range(50,400):
        #step1. 获取高频无效词列表，为了防止影响效果
        invalid_words = read_invalid_words(data_path,top_k=invalid_k) 
        
        # 获取所有类别下的高频词，组合成一个feature_words，词频
        for valid_k in range(100,500):
            feature_words = get_high_freq_word_in_class(data_path = train_data_path,top_k = valid_k,invalid_words = invalid_words)
            
            # step2. 读取数据集
            train_data,labels = read_train_data(train_data_path)
            # 先划分train + dev
            x_train,x_dev,y_train,y_dev = train_test_split(train_data,labels,test_size=0.3,random_state=32,shuffle=True)
            
            # step3.根据特征集合，找出每个训练文本的特征
            # 找出train,dev,test 的特征
            train_feature_list  = text_features(x_train,feature_words)
            dev_feature_list = text_features(x_dev,feature_words)
            # test 数据
            test_data = read_test_data(test_data_path)
            test_feature_list = text_features(test_data,feature_words)
            # step4.根据特征进行分类
            classifier,dev_macro_f1 = text_classifier(train_feature_list, dev_feature_list,y_train,y_dev)
            
            if dev_macro_f1 > max_dev_score:
                max_dev_score = dev_macro_f1
                print(f"invalid_k={invalid_k},valid_k = {valid_k},dev_macro_f1={dev_macro_f1},",)
                res = classifier.predict(test_feature_list)
                # print(res)
                label = [id2label[i] for i in res]
                ids = [i for i in range(len(test_data))]
                res = pd.DataFrame({'id':ids,
                                    'label':label})
                submit_path = "submission_nb.csv"
                res.to_csv(submit_path,index=False)
