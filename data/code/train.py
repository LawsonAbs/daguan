import collections
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import f1_score
from visdom import Visdom
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json
import time
import pickle
import random
import warnings
import numpy as np
from torch.optim import Optimizer
from collections import defaultdict
from lookahead import Lookahead
from torch.nn import BCELoss
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertTokenizerFast
)
from modeling.modeling_nezha.modeling import NeZhaPreTrainedModel, NeZhaModel
from modeling.bert.modeling_bert import BertModel, BertPreTrainedModel
from sparse_max.sparsemax import SparsemaxLoss
from sklearn.model_selection import StratifiedKFold,KFold

label_dict = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}

            
label_list = ['5-24', '6-34', '1-1', '6-8', '10-26', '2-3', '5-22', '6-28', '8-18', '1-4', '2-6', '6-21', '7-16', '6-29', '6-20', 
              '6-15', '6-13', '9-23', '5-35', '2-33', '5-30', '1-9', '8-27', '1-10', '6-19', '3-5', '2-2', '4-7', '2-17', '5-12', 
              '6-32', '6-31', '2-25', '2-11', '2-14']



viz = Visdom()
win_1 = "0.36_train_loss"
win_2= "0.36_macro_f1_eval"
win_3 = "3-5 loss_2"
opts = {
    "xlabel":'step',
    "ylabel":"value",
    "title":"0.36 loss"
}


# ?????????train - dev 
# ????????????????????????????????????????????????????????? sklearn.model_selection ?????? StratifiedKFold ????????? => 
# ?????????????????????????????????????????????????????????????????????
def split_data_by_class(x,y,rate,seed=22):
    '''
    x,y ????????????????????????????????????
    ??????random = True, ???????????????~
    rate??????dev????????????????????????????????????1????????????1?????????

    returns:
        train_idx,dev_idx
    '''    
    # ???shuffle??????????????????????????????shuffle  
    random.seed(seed)
    random.shuffle(y)
    random.seed(seed)
    random.shuffle(x)

    cont_id = {} # ???????????????????????????list???    
    # ??????????????????????????????
    for i in range(len(y)):
        y_idx = y[i] # y_idx???????????????
        if y_idx not in cont_id.keys():
            cont_id[y_idx] = []
        cont_id[y_idx].append(i) # ?????????????????????????????????
    
    train_idx,dev_idx = [],[] # ????????????????????????????????????
    # ??????????????????????????????
    for item in cont_id.items():
        key,value = item # key ??????????????????value???????????????????????????y????????????
        if len(value) >= 2: # ????????????????????????????????????2????????????dev
            mid = int(len(value)*rate)
            if mid == 0: # ????????????0???????????????????????????1
                mid = 1
            for i in range(mid): # ?????????????????? rate * len(value) ???
                dev_idx.append(value[i])         
            for i in range(mid,len(value)): # ???????????????????????????train ??????
                train_idx.append(value[i])
        else:            
             train_idx.extend(value) # ?????????????????????train
    
    # ??????????????????shuffle?????????????????????????????????
    random.seed(seed)
    random.shuffle(train_idx)
    random.seed(seed)
    random.shuffle(dev_idx)
    
    # ?????????????????????????????????dev_idx ??????????????????????????????????????????????????????????????????????????????????????????
    while len(dev_idx) < len(train_idx) * rate:
        idx = random.choice(train_idx)        
        dev_idx.append(idx) # ??????dev???
        train_idx.remove(idx) # ????????????idx

    return train_idx,dev_idx # ??????train/dev ?????????????????????


class NeZhaSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 35
        # self.num_labels = 20
        self.bert = BertModel(config) 
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.multi_drop = 5
        self.multi_dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(self.multi_drop)])        
        self.loss_fct = LabelSmoothingLoss(smoothing=0.01)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        for j, dropout in enumerate(self.multi_dropouts):
            if j == 0:
                logits = self.classifier(dropout(pooled_output)) / self.multi_drop
            else:
                logits += self.classifier(dropout(pooled_output)) / self.multi_drop
        outputs = (logits,) + outputs[2:]

        
        if labels is not None:            
            # loss_fct = nn.CrossEntropyLoss()
            # loss_fct = SparsemaxLoss()
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
            # print("?????????????????????????????????",input_ids)
            # print("??????????????????",labels)
            # print("????????????",loss)
        return outputs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

# ??????for??????????????????yield?????????????????????????????????????????????????????????
# ?????????????????????????????????????????????batch???????????????????????????????????????
def batch_loader(config, src, tgt, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        tgt_batch = tgt[ins_num // batch_size * batch_size:]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch

def read_single_dataset(config, tokenizer, path):
    start = time.time()
    dataset = []
    seq_length = config['max_seq_len']
    label_list = ['?????????', '????????????', '????????????']
    print('>> load data:', config[path])
    with open(config[path], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sent_a, sent_b, tgt = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
            src = src_a + src_b
            seg = [0] * len(src_a) + [1] * len(src_b)
            mask = [1] * len(src)
            tgt = int(tgt)
            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, tgt, seg, mask))
    return dataset

def read_dataset(config, tokenizer):
    start = time.time()
    dataset = []
    seq_length = config['max_seq_len']
    print('>> load data:', config['data_path'])
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            id, sent_a, tgt = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src = src_a
            seg = [0] * len(src_a)
            mask = [1] * len(src)
            tgt = int(tgt)
            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, tgt, seg, mask))
    
    # data_cache_path = config['normal_data_cache_path']
    # if not os.path.exists(os.path.dirname(data_cache_path)):
    #     os.makedirs(os.path.dirname(data_cache_path))
    # with open(data_cache_path, 'wb') as f:
    #     pickle.dump(dataset, f)
    
    print("\n>> loading sentences from {},Time cost:{:.2f}".
          format(config['data_path'], ((time.time() - start) / 60.00)))

    return dataset


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    weight = t.ones[]
    """
    # ?????????smoothing ??????????????????????????????0.01 ????????????
    def __init__(self, smoothing=0.01,weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.step = 0

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        # step1. ??????NLLLoss????????????CrossEntropy ????????????
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        
        # ??????weight????????????
        nll_loss = nll_loss.squeeze(1)
        a = nll_loss.tolist()
        # print(f"?????????????????????????????????nll_loss={a}")
        temp = [id2label[i] for i in target.tolist()]
        # print(f"?????????????????????????????????label = {temp}")
        # step2. => ?????????
        smooth_loss = -log_probs.mean(dim=-1)
        if '3-5' in temp:
            idx = temp.index('3-5')
            special_loss = nll_loss[idx].item()
            viz.line( [special_loss],[self.step],win=win_3,update='append')
            self.step += 1
        # ???????????????self.confidence * nll_loss?
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# ??????????????????
class FGM:
    def __init__(self, config, model):
        self.model = model
        self.backup = {}
        self.emb_name = config['emb_name']
        self.epsilon = config['epsilon']

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, config, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = config['epsilon']
        self.emb_name = config['emb_name']
        self.alpha = config['alpha']

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def block_shuffle(config, train_set):
    bs = config['batch_size'] * 100 # block ??????
    num_block = int(len(train_set) / bs) # ???????????????block??????train data ??????
    slice_ = num_block * bs
    # ???????????????????????????????????????????????????????????????????????????
    train_set_to_shuffle = train_set[:slice_]
    train_set_left = train_set[slice_:]

    # ?????????????????????????????????????????????tokenizer ???????????????padding 
    sorted_train_set = sorted(train_set_to_shuffle, key=lambda i: len(i[0]))
    shuffled_train_set = []

    # ????????????????????????temp????????????????????????temp??? => ???????????????shuffle????????????????????????batch????????????
    tmp = []
    for i in range(len(sorted_train_set)):
        tmp.append(sorted_train_set[i])
        if (i+1) % bs == 0:
            random.shuffle(tmp)
            shuffled_train_set.extend(tmp)
            tmp = []

    # ??????????????????shuffle 
    random.shuffle(train_set_left)
    shuffled_train_set.extend(train_set_left)

    return shuffled_train_set


def build_model_and_tokenizer(config):
    tokenizer_path = config['model_path'] + '/vocab.txt'
    if config['tokenizer_fast']:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if config['use_model'] == 'nezha':
        model = NeZhaSequenceClassification.from_pretrained(config['model_path'])
    # if config['use_model'] == 'bert':
    #     model = BertModel.from_pretrained(config['model_path'])
    return tokenizer, model


# ?????????????????????????????????????????????macro_f1 ???
def cal_f1(preds, trues):
    macro_f1 = f1_score(trues,preds,average='macro')    
    return macro_f1

def build_optimizer(config, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': config['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=1e-8)
    optimizer = Lookahead(optimizer, k=3, alpha=0.5)#TODO ??????????????????
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * config['warmup_ratio'],
                                     t_total=train_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    return optimizer, scheduler


def train():
    config = {
        'use_model': 'nezha',
        'normal_data_cache_path': '',  # ?????????????????? ??????????????????
        'data_path': '../user_data/train_balance_0.36_repeat.txt', # ????????????
        'output_path': '../user_data/fine_tune_model/model', # fine-tuning????????????????????????
        'model_path': '../pretrain_model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000', # your pretrain model path
        'shuffle_way': 'block_shuffle',  # block_shuffle ?????? random shuffle
        'use_swa': True, # ???????????????????????????
        'tokenizer_fast': False,
        'batch_size': 8,
        'num_epochs': 10,
        'max_seq_len': 100,
        'learning_rate': 2e-5,
        'alpha': 0.3,  # PGD???alpha???????????? 
        'epsilon': 1.0, # FGM???epsilon???????????? 
        'adv_k': 3, # PGD???????????????
        'emb_name': 'word_embeddings.', 
        'adv': 'fgm', # ??????????????????
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 50, # ???500?????????logger
        'seed': 124525601, # ????????????
        'fold': 5, # k-flod  => ???????????????0.2 ???????????????dev???
        'dev': False # ?????????????????????
        }
    # ??????pretrain_model_name
    pretrain_model_name = config["model_path"].split("/")[-1]
    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))
    
    tokenizer, model = build_model_and_tokenizer(config)
    # if not os.path.exists(config['normal_data_cache_path']):
    #     train_set = read_dataset(config, tokenizer)
    #     # train_set = read_single_dataset(config, tokenizer, 'data_path2')
    # else:
    #     with open(config['normal_data_cache_path'], 'rb') as f:
    #         train_set = pickle.load(f)
    # ??????????????????????????????????????????
    train_set = read_dataset(config, tokenizer)
    seed_everything(config['seed'])

    if config['shuffle_way'] == 'block_shuffle':
        train_set = block_shuffle(config, train_set)
    else:
        random.shuffle(train_set)  # ??????shuffle????????????

    train_num = len(train_set)

    # +1 ?????????????????? range() ????????????
    train_steps = int(train_num * config['num_epochs'] / config['batch_size']) + 1
    
    optimizer, scheduler = build_optimizer(config, model, train_steps)
    model.to(config['device'])    
    src = [example[0] for example in train_set]
    y = [example[1] for example in train_set]
    seg = [example[2] for example in train_set]
    mask = [example[3] for example in train_set]
    
    # train_data = TensorDataset(src, tgt, seg, mask)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['batch_size'], num_workers=4)
    cudnn.benchmark = True
    
    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    start = time.time()
    
    # ????????????StratifiedKFold ?????????
    # skf = StratifiedKFold(n_splits=config['fold'],shuffle=True,random_state=config['seed'])    
    x = [] 
    for input_ids, type_ids, attention_masks in zip(src, seg, mask):
        x.append((input_ids, type_ids, attention_masks))    
    
    # ????????????????????????[X_train,y_train]????????????????????????????????????????????????????????????random_state ????????????????????????
    # X?????????????????????y?????????????????????train????????????????????????test??????????????????
    # ??????????????? train_test_split() ???????????????????????????????????????????????????????????????  
    # X_train,X_test,y_train,y_test = train_test_split(kfold_dataset,tgt_numpy,test_size=0.15,random_state=22)
    
    x = np.array(x)
    y = np.array(y)
    # tgt_numpy ???????????????????????????shuffle??????????????????????????????????????????x,y??????????????????????????????shuffle???
    if config['dev']:
        train_idx, dev_idx = split_data_by_class(x,y,rate=0.2,seed=config['seed'])
        # ??????numpy?????????        
        x_train , x_test = x[train_idx],x[dev_idx]
        y_train ,y_test = y[train_idx],y[dev_idx]
        
        # train
        src = torch.LongTensor([example[0] for example in x_train])
        seg = torch.LongTensor([example[1] for example in x_train])
        mask = torch.LongTensor([example[2] for example in x_train])
        tgt = torch.LongTensor(y_train)        
        
        # eval
        eval_src = torch.LongTensor([example[0] for example in x_test])
        eval_seg = torch.LongTensor([example[1] for example in x_test])
        eval_mask = torch.LongTensor([example[2] for example in x_test])
        eval_tgt = torch.LongTensor(y_test)   
        
    else:        
        src = torch.LongTensor([example[0] for example in x])
        seg = torch.LongTensor([example[1] for example in x])
        mask = torch.LongTensor([example[2] for example in x])
        tgt = torch.LongTensor(y) 
     

    # start train !!
    total_loss = 0.0
    global_steps = 0
    for epoch in range(1, config['num_epochs'] + 1):
        cur_avg_loss = 0.0
        model.train()        
        i = 1
        for (src_batch, tgt_batch, seg_batch, mask_batch) \
                in tqdm(batch_loader(config, src, tgt, seg, mask)):
            src_batch = src_batch.to(config['device'])
            tgt_batch = tgt_batch.to(config['device'])
            seg_batch = seg_batch.to(config['device'])
            mask_batch = mask_batch.to(config['device'])

            # lookahead.zero_grad()
            output = model(input_ids=src_batch, labels=tgt_batch,
                        token_type_ids=seg_batch, attention_mask=mask_batch)
            loss = output[0]
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            cur_avg_loss += loss.item()

            # ?????????????????????????????????????????????
            if config['adv'] == 'fgm':
                fgm = FGM(config, model)
                fgm.attack()
                adv_loss = model(input_ids=src_batch, labels=tgt_batch,
                                token_type_ids=seg_batch, attention_mask=mask_batch)[0]
                adv_loss.backward()
                fgm.restore()

            if config['adv'] == 'pgd':
                pgd = PGD(config, model)
                K = config['adv_k']
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss = model(input_ids=src_batch, labels=tgt_batch,
                                    token_type_ids=seg_batch, attention_mask=mask_batch)[0]
                    adv_loss.backward()
                pgd.restore()
            optimizer.step()
            # lookahead.step()
            # if epoch > swa_start:
            #     swa_model.update_parameters(model)
            #     swa_scheduler.step() 
            # else:
            #     scheduler.step()
            scheduler.step()
            model.zero_grad()

            # ??????loss????????? 
            if (i + 1) % config['logging_step'] == 0:
                print("\n>> epoch - {}, epoch steps - {}, global steps - {}, "
                    "epoch avg loss - {:.4f}, global avg loss - {:.4f}, time cost - {:.2f} min".format
                    (epoch, i + 1, global_steps + 1, cur_avg_loss / config['logging_step'],
                    total_loss / (global_steps + 1),
                    (time.time() - start) / 60.00))
                viz.line([cur_avg_loss / config['logging_step']],[global_steps],win=win_1,update='append')
                cur_avg_loss = 0.0
            i +=1
            global_steps += 1
        
        if config['dev']:
            # train??????????????????????????????????????????????????????
            avg_loss ,all_label = [],[] # ????????????precision,recall ???????????????
            best_f1 = 0 # ???????????????f1 ???
            model.eval()
            # ?????????????????????batch?????????????????????all_preds??????????????????????????????        
            all_preds = [] 
            for i, (src_batch,tgt_batch, seg_batch, mask_batch, ) in enumerate(tqdm(batch_loader(config, eval_src, eval_tgt, eval_seg, eval_mask))):
                src_batch = src_batch.to(config['device'])
                tgt_batch = tgt_batch.to(config['device'])
                seg_batch = seg_batch.to(config['device'])
                mask_batch = mask_batch.to(config['device'])
                with torch.no_grad():
                    output = model(input_ids=src_batch, labels=tgt_batch,
                                    token_type_ids=seg_batch, attention_mask=mask_batch)
                loss = output[0]
                avg_loss.append(loss.item())
                logits = torch.softmax(output[1], 1)            
                preds = torch.argmax(logits,-1)
                all_preds.extend(preds.tolist())    
                all_label.extend(tgt_batch.tolist())

            # ?????? f1_score ??????????????? macro_f1 ???
            macro_f1 = cal_f1(all_preds, all_label)      
            print(classification_report(all_label,all_preds,
                                        target_names=label_list                                        
                                        )
                                        ) # ??????????????????
            print("macro_f1 = ",macro_f1)
            viz.line([macro_f1],[epoch],win=win_2,update='append')
            if macro_f1 > best_f1 :# ?????????????????????????????????
                best_f1 = macro_f1
        macro_f1 = 0
        model_save_path = os.path.join(config['output_path'], f'{pretrain_model_name}_epoch_{epoch}_repeat_0.36')
        # if os.path.exists(model_save_path):
        #     os.remove(model_save_path)
        print('model_save_path:', model_save_path)
        # hasattr ??????????????????????????????????????????????????????true???????????????false
        model_to_save = model.module if hasattr(model, 'module') else model
        print('\n>> model saved ... ...')
        model_to_save.save_pretrained(model_save_path)
        conf = json.dumps(config)
        out_conf_path = os.path.join(config['output_path'], f'{pretrain_model_name}_epoch_{epoch}_repeat_0.36' +
                                    '/train_config.json')
        with open(out_conf_path, 'w', encoding='utf-8') as f:
            f.write(conf)
            # else:
            #     early_stopping += 1
            #     print(f"Counter {early_stopping} of {config['early_stopping']}")
            #     if early_stopping > config['early_stopping']:
            #         print("Early stopping with best_f1: ", best_f1, "and val_f1 for this epoch: ", avg_f1, "...")
            #         break
        
    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))


if __name__ == '__main__':
    train()