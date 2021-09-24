###
 # @Author: LawsonAbs
 # @Date: 2021-09-22 23:32:51
 # @LastEditTime: 2021-09-24 19:53:36
 # @FilePath: /daguan_gitee/code/run.sh
### 

# 1.处理数据
python process_data.py # 处理训练数据和测试数据

# 生成预训练使用的词表，会生成两个词表vocab_1.txt 和 vocab_2.txt 
# vocab_1.txt 考虑到词频信息，只有7k+单词； 而vocab_2.txt 有2.1w+ 单词
python build_vocab.py 


# 2.预训练


# 3.微调
python train.py


# 4.预测
## 4.1先对少样本数据预测
python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000_epoch_10_repeat_0.34

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000_epoch_10_repeat_0.36

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.4M_large_40000_128_checkpoint-10_0.4

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.4M_large_40000_128_checkpoint-10_0.6

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000_checkpoint-10_ls_0.01


## 4.2再对所有样本数据预测
python predict.py --load_model_path ../user_data/fine_tune_model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000_checkpoint-10_ls_0.01

python predict.py --load_model_path ../user_data/fine_tune_model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000_epoch_10_augment

python predict.py --load_model_path ../user_data/fine_tune_model/checkpoint_0.589


# 5.Ensemble & Combine
python code/utils/tools.py