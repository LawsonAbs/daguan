###
 # @Author: LawsonAbs
 # @Date: 2021-09-22 23:32:51
 # @LastEditTime: 2021-09-24 21:35:10
 # @FilePath: /daguan_gitee/code/run.sh
### 

# 1.处理数据
python process_data.py # 处理训练数据和测试数据

# 生成预训练使用的词表，会生成两个词表vocab_1.txt 和 vocab_2.txt 
# vocab_1.txt 考虑到词频信息，只有7k+单词； 而vocab_2.txt 有2.1w+ 单词
python build_vocab.py 


# 2.预训练
# echo ">>>>>>>>>>>> 开始使用模型 bert-base-chinese 进行预训练。 <<<<<<<<<\n\n"
# cd pretrain
#============================= step 1. 使用 bert-base-chines 预训练 =============================
# python pretrain.py --vocab_file ../../user_data/data/vocab_1.txt  \
#                    --train_file_path ../../user_data/data/all.txt  \
#                    --model_name   bert-base-chinese \
#                    --num_epoch 200
# echo ">>>>>>>>>>>> 在bert-base-chinese基础上的预训练结束！<<<<<<<<<\n\n"

#============================= step 2. 使用 chinese-roberta-wwm-ext-large 预训练 =============================
# 以vocab_2.txt 在large 上训练
# 先在 2.4G 的数据上预训练
# echo ">>>>>>>>>>>> 在chinese-roberta-wwm-ext-large 基础上进行预训练 <<<<<<<<<\n\n"
# python pretrain.py --vocab_file ../../user_data/data/vocab_2.txt  \
#                    --train_file_path ../../user_data/data/1.txt  \
#                    --model_name chinese-roberta-wwm-ext-large  \
#                    --num_epoch 100

# 再在 all.txt（train.txt+test.txt） 上预训练，训练200 epoch
# python pretrain.py --vocab_file ../../user_data/data/vocab_2.txt  \
#                    --train_file_path ../../user_data/data/all.txt  \
#                    --model_name   ../../user_data/pretrain_model/checkpoint_epoch_100_large \
#                    --num_epoch 200
# echo ">>>>>>>>>>>> 在 chinese-roberta-wwm-ext-large 基础上的预训练结束！<<<<<<<<<\n\n"

# 3.微调
# python train.py


# 4.预测
## 4.1先对少样本数据预测
# 如下几个文件的预训练模型都存放在百度云链接中
python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/A_epoch_10_repeat_0.34

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/A_epoch_10_repeat_0.36

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.4M_large_40000_128_checkpoint-10_0.4

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/2.4G+4.4M_large_40000_128_checkpoint-10_0.6

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/A_epoch_10_0.592


## 4.2再对所有样本数据预测
python predict.py --load_model_path ../user_data/fine_tune_model/A_epoch_10_0.592

python predict.py --load_model_path ../user_data/fine_tune_model/A_epoch_10_augment_0.580

python predict.py --load_model_path ../user_data/fine_tune_model/C_epoch_10_0.589


# 5.Ensemble & Combine
python code/tools.py