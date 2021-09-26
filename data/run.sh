###
 # @Author: LawsonAbs
 # @Date: 2021-09-22 23:32:51
 # @LastEditTime: 2021-09-26 15:58:01
 # @FilePath: /daguan_gitee/data/run.sh
### 

mkdir prediction_result/less/
mkdir prediction_result/normal/

cd code # 进入到代码目录
# 1.处理数据
python process_data.py # 处理训练数据和测试数据

# 生成预训练使用的词表，会生成两个词表vocab_1.txt 和 vocab_2.txt 
# vocab_1.txt 考虑到词频信息，只有7k+单词； 而vocab_2.txt 有2.1w+ 单词
# python build_vocab.py 


# 2.预训练
# echo ">>>>>>>>>>>> 开始使用模型 bert-base-chinese 进行预训练。 <<<<<<<<<\n\n"
# cd pretrain
#============================= step 1. 使用 bert-base-chines 预训练 =============================
# python pretrain.py --vocab_file user_data/data/vocab_1.txt  \
#                    --train_file_path user_data/data/all.txt  \
#                    --model_name   bert-base-chinese \
#                    --num_epoch 200
# echo ">>>>>>>>>>>> 在bert-base-chinese基础上的预训练结束！<<<<<<<<<\n\n"

#============================= step 2. 使用 chinese-roberta-wwm-ext-large 预训练 =============================
# 以vocab_2.txt 在large 上训练
# 先在 2.4G 的数据上预训练
# echo ">>>>>>>>>>>> 在chinese-roberta-wwm-ext-large 基础上进行预训练 <<<<<<<<<\n\n"
# python pretrain.py --vocab_file user_data/data/vocab_2.txt  \
#                    --train_file_path user_data/data/1.txt  \
#                    --model_name chinese-roberta-wwm-ext-large  \
#                    --num_epoch 100

# 再在 all.txt（train.txt+test.txt） 上预训练，训练200 epoch
# python pretrain.py --vocab_file user_data/data/vocab_2.txt  \
#                    --train_file_path user_data/data/all.txt  \
#                    --model_name   user_data/pretrain_model/checkpoint_epoch_100_large \
#                    --num_epoch 200
# echo ">>>>>>>>>>>> 在 chinese-roberta-wwm-ext-large 基础上的预训练结束！<<<<<<<<<\n\n"

# 3.微调
# python train.py


# 4.预测
## 4.1先对少样本数据预测
python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/less/A_epoch_10_repeat_0.34_less

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/less/A_epoch_10_repeat_0.36_less

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/less/A_epoch_7_repeat_0.34_less

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/less/B_checkpoint-10_0.4_less

python predict.py --less=True \
                  --load_model_path ../user_data/fine_tune_model/less/B_checkpoint-10_0.5_less

# 对移动文件操作的解释请主办方见README.md 中的5.3，如果不移动如下文件，则将下面6行（line 68-75）的注释取消，用于代替生成此文件
mv submission/submission_balance_10_num_30.csv ../prediction_result/less/
mv submission/submission_balance_10_num_60.csv ../prediction_result/less/
# python predict.py --less=True \
#                 --load_model_path ../user_data/fine_tune_model/less/D_replace_60_less \
#                 --max_seq_len 100
# python predict.py --less=True \
#                 --load_model_path ../user_data/fine_tune_model/less/D_replace_90_less \
#                 --max_seq_len 100


## 4.2再对所有样本数据预测（normal）
python predict.py --load_model_path ../user_data/fine_tune_model/normal/A_epoch_10_0.592

python predict.py --load_model_path ../user_data/fine_tune_model/normal/A_epoch_10_augment_0.580

# 因为这个 C_epoch_10_0.589是一个较早期训练的模型，使用的是max_seq_len 是300，但是后面分析这个参数设置成128就足矣了，
# 但是因为要完全复现，所以这里仍然使用这个看上去设置不佳的参数。
python predict.py --load_model_path ../user_data/fine_tune_model/normal/C_epoch_10_0.589 \
                --max_seq_len 300 

# 对移动文件操作的解释请见README.md 中的5.3
mv submission/submission_0905_0.583.csv ../prediction_result/normal/
#  python predict.py --load_model_path ../user_data/fine_tune_model/normal/D_replace_0.583 \
#                 --max_seq_len 100 


# # 5.Ensemble & Combine
python tools.py