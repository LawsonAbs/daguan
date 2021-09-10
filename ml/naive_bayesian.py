#coding: utf-8
import json
import re
import os
import time
import random
import nltk
import sklearn
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt


# 从文本中挑选出无用的词
# 使用unlabeled 的数据集，基于大数据分析查找出各个词的出现频率，同普通文本进行对比
def get_invalid_words(data_path):
    invalid_words = []
    word_cnt = {} # word => cnt 的dict
    word_freq = {} # word => freq 的dict
    file_list = os.listdir(data_path)
    total = 0 
    for file in file_list: # 遍历取出所有文件
        with open(file,'r') as f:
            cont = f.read() # 读取所有内容
            cont = re.split(r'[，。！？-——*%（） ]')
            print(cont) # 查看得到的文本内容
            for word in cont:
                if word not in word_cnt.keys(): # 判断它得是一个单词
                    word_cnt[word] = 1
                else:
                    word_cnt[word] += 1
                total += 1

    for item in word_cnt.items():
        word,cnt = item
        freq = cnt/total
        word_freq[word] = freq
    word_freq = sorted(word_freq.items(),key = lambda x:x[1],reverse=True) # 按照频率排序
    word_freq_path = "/home/lawson/program/data/word_freq.json"
    with open(word_freq_path,'w' ) as f:
        json.dump(f)
        
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

# 
def TextProcessing(folder_path, test_size=0.2):
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
    # train_data_list, test_data_list, train_class_list, test_class_list = sklearn.cross_validation.train_test_split(data_list, class_list, test_size=test_size)
    data_class_list = zip(data_list, class_list)
    random.shuffle(data_class_list)
    index = int(len(data_class_list)*test_size)+1
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)

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

    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


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


def TextFeatures(train_data_list, test_data_list, feature_words, flag='nltk'):
    def text_features(text, feature_words):
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
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag='nltk'):
    ## -----------------------------------------------------------------------------------
    if flag == 'nltk':
        ## nltk分类器
        train_flist = zip(train_feature_list, train_class_list)
        test_flist = zip(test_feature_list, test_class_list)
        classifier = nltk.classify.NaiveBayesClassifier.train(train_flist)
        # print classifier.classify_many(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.classify(test_feature),
        # print ''
        test_accuracy = nltk.classify.accuracy(classifier, test_flist)
    elif flag == 'sklearn':
        ## sklearn分类器
        classifier = MultinomialNB().fit(train_feature_list, train_class_list)
        # print classifier.predict(test_feature_list)
        # for test_feature in test_feature_list:
        #     print classifier.predict(test_feature)[0],
        # print ''
        test_accuracy = classifier.score(test_feature_list, test_class_list)
    else:
        test_accuracy = []
    return test_accuracy


if __name__ == '__main__':
    ## 文本预处理
    folder_path = './Database/SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    # 生成stopwords_set
    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    ## 文本特征提取和分类
    # flag = 'nltk'
    flag = 'sklearn'
    deleteNs = range(0, 1000, 20)
    test_accuracy_list = []
    for deleteN in deleteNs:
        # feature_words = words_dict(all_words_list, deleteN)
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words, flag)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list, flag)
        test_accuracy_list.append(test_accuracy)
    print(test_accuracy_list)

    # 结果评价
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.savefig('result.png')
    print("end")