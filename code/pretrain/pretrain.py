# coding:utf-8
import re
from re import M
import sys
from packaging.version import parse
from transformers.models import bert
import os
import pickle
import torch
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict
from torch.utils.data import Dataset
from transformers import BertForMaskedLM
from transformers import (
    BertTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizer, BertConfig
)
from transformers.utils import logging

import sys
sys.path.append(r"../../code")  # 引入当前的这个code下的环境
print(sys.path)
from modeling.modeling_nezha.modeling import NeZhaForMaskedLM,NeZhaConfig
from simple_trainer import Trainer
from pretrain_args import ParallelMode, TrainingArguments


import time
warnings.filterwarnings('ignore')
from argparse import ArgumentParser
logger = logging.get_logger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def read_data(config, train_file_path, test_file_path, tokenizer: BertTokenizer) -> dict:
    train_df = pd.read_csv(train_file_path, header=None, sep='\t')
    test_df = pd.read_csv(test_file_path, header=None, sep='\t')

    pretrain_df = pd.concat([train_df, test_df], axis=0)

    inputs = defaultdict(list)
    for i, row in tqdm(pretrain_df.iterrows(), desc=f'preprocessing pretrain data ... ...', total=len(pretrain_df)):
        sentence_a, sentence_b = row[0], row[1]
        inputs_dict = tokenizer.encode_plus(sentence_a, sentence_b, add_special_tokens=True,
                                            return_token_type_ids=True, return_attention_mask=True)
        inputs['input_ids'].append(inputs_dict['input_ids'])
        inputs['token_type_ids'].append(inputs_dict['token_type_ids'])
        inputs['attention_mask'].append(inputs_dict['attention_mask'])

    data_cache_path = config['data_cache_path']

    if not os.path.exists(os.path.dirname(data_cache_path)):
        os.makedirs(os.path.dirname(data_cache_path))
    with open(data_cache_path, 'wb') as f:
        pickle.dump(inputs, f)

    return inputs


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer,
     train_file_path: str, 
     block_size: int,
     vocab_path
     ):
        assert os.path.isfile(train_file_path), f"Input file path {train_file_path} not found"
    
        print(f"Creating features from dataset file at {train_file_path}")
        batch_encoding = []
        input_ids = []
        vocab_map = {} # word => id
        index = 0
        # 写一个获取vocab映射的
        with open(vocab_path, 'r', encoding='utf-8') as f1:
            for line in f1:            
                line = line.strip("\n")
                vocab_map[line] = index
                index += 1

        # 字典中数到id的映射关系是一一对应        
        with open(train_file_path, encoding="utf-8",errors ='ignore') as f:
            # isspace 用于判断一个字符串中的字符是否全是whitespace                    
            flag  = 0
            for line in tqdm(f): # tqdm中可以加total参数表示总行数
                temp_input_ids = [0] * block_size
                temp_input_ids[0] = 2
                if len(line )>0 and not line.isspace():
                    line = line.strip("\n")                    
                    row = re.split(r'([，。？！ ])',line)
                    max_length = block_size # 最大长度
                    cnt = 1
                    for i in row:
                        if i ==' ' or i =='':
                            continue
                        if i not in vocab_map.keys():
                            flag +=1
                            temp_input_ids[cnt] = 1 # unknown
                        else:
                            temp_input_ids[cnt] = vocab_map[i]
                        if cnt >= max_length - 1:
                            break
                        cnt +=1                
                temp_input_ids[-1] = 3
                if (len (temp_input_ids)==block_size):                    
                    input_ids.append(temp_input_ids) # 放入到所有的当中

        # with open(train_file_path, encoding="utf-8") as f:
        #     train_lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        # # 不能在这里是tokenizer，否则很费时间
        # # batch_encoding = tokenizer(train_lines, add_special_tokens=True, truncation=True, max_length=block_size)
        # batch_encoding = tokenizer(train_lines, truncation=True, max_length=block_size,padding='max_length')

        self.examples = input_ids
        #self.examples = batch_encoding['input_ids']
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]
        print(flag)
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def main():
    """
    download pretrain model from https://github.com/lonePatient/NeZha_Chinese_PyTorch,
    we only use pretrain model name : nezha-cn-base, nezha-base-wwm
    """
    parser = ArgumentParser()
    parser.add_argument("--pretrain_type", default=None,type=str)
    parser.add_argument("--manual_seed", default=123456, type=int,help='seed num')
    parser.add_argument("--train_fgm", default=False,type=boolean_string)
    parser.add_argument("--fgm_epsilon", default=1.0)
    parser.add_argument("--batch_size", default=8,type=int)
    parser.add_argument("--num_epochs",default=100,type=int)
    parser.add_argument("--gradient_accumulation_steps", default=2,type=int)        
    parser.add_argument("--data_cache_path,",default='', type=str)
    parser.add_argument("--seq_length", default=50, type=int)
    parser.add_argument("--model_name", default="bert-base-chinese", type=str) # 使用nezha 预训练还是使用 large预训练，这个参数必须指定
    parser.add_argument("--model_type",default="bert", type=str)
    parser.add_argument("--model_save_path",default="", type=str)
    
    
    parser.add_argument("--model_path",default="",type=str) 
    parser.add_argument("--vocab_file",default="../../user_data/data/vocab_1.txt",type=str) # 必须指定，否则不知道是用vocab_1.txt 还是 vocab_2.txt
    parser.add_argument("--train_file_path",default="../../user_data/data/1.txt",type=str)
    parser.add_argument("--config_path",default="config.json",type=str) 
    config = parser.parse_args()

    mlm_probability = 0.15
    num_train_epochs = config.num_epochs
    seq_length = 50
    batch_size = config.batch_size
    fgm_epsilon = 1.0
    learning_rate = 2e-5
    save_steps = 1000 
    seed = config.manual_seed
    
    if 'fgm' in config.model_save_path:
        use_fgm = False
    else:
        use_fgm = False
    
    model_save_path = config.model_save_path
    gradient_accumulation_steps = config.gradient_accumulation_steps
    print(config)
    print(f'use_fgm={use_fgm}')
        
    model_path = "../../user_data/init_model/" + config.model_name + "/pytorch_model.bin"
    config_path = "../../user_data/init_model/"+config.model_name +"/"+ config.config_path
    vocab_file = config.vocab_file
    
    tokenizer = BertTokenizer.from_pretrained(vocab_file)

    assert os.path.isfile(model_path), f"Input file path {model_path} not found, " \
                                       f"please download relative pretrain model in huggingface or" \
                                       f"https://github.com/lonePatient/NeZha_Chinese_PyTorch " \
                                       f"model name:nezha-cn-base or nezha-base-wwm"
    if 'wwm' in config.model_save_path:
        print('>> Whole Word Mask ...')
        data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                                     mlm=True,
                                                     mlm_probability=mlm_probability)
    else:
        print('>> Mask Language Model ...')
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                        mlm=True,
                                                        mlm_probability=mlm_probability)             

    if config.model_type == 'nezha':
        model_config = NeZhaConfig.from_pretrained(config_path)
        model = NeZhaForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                          config=model_config)

    if config.model_type == 'bert':
        model_config = BertConfig.from_pretrained(config_path)
        # 下面这种方式就是随机初始化的
        # model = BertForMaskedLM(config=model_config)
        # model = BertForMaskedLM.from_pretrained("/home/lawson/program/daguan/pretrain_model//final")
        model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 config=model_config)

    if use_fgm:
        print('进行fgm预训练')
        #model = FGMWrapper(model, epsilon=fgm_epsilon)
    
    print('>> train data load start....')
    print(f"训练使用的参数如下：")
    temp = vars(config) 
    for item in temp.items():
        print(item)
    start_time = time.time()
    training_args = TrainingArguments(
            output_dir='../../user_data/pretrain_model/'+config.model_path,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size, # 注意这里是每个设备上的batch_size，而不是总共的batch_size 
            # save_strategy="epoch", # 使用epoch作为最后的保存模型，也可以使用step
            save_steps=save_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=500,
            save_total_limit=5,
            prediction_loss_only=True,
            seed=seed,
            # batch_size= 8
        )

    
    train_file_path = config.train_file_path
    dataset = Dataset()
    dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                    train_file_path=train_file_path,                                        
                                    block_size=seq_length,
                                    vocab_path =vocab_file
                                    )
    print('>> train data load end....')
    print('>> load data cost {} s'.format(time.time()- start_time))
    
    print('start train....')
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    
    trainer.train()
    trainer.save_model(model_save_path)
    
if __name__ == '__main__':
    main()