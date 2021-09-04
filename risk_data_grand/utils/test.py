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
