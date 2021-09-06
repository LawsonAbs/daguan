'''
Author: LawsonAbs
Date: 2021-09-04 22:07:40
LastEditTime: 2021-09-05 19:37:39
FilePath: /daguan/risk_data_grand/utils/test.py
'''
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


if __name__ == '__main__':
    get_vocab_map("")
    