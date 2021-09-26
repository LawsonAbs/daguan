###
 # @Author: LawsonAbs
 # @Date: 2021-09-22 23:32:51
 # @LastEditTime: 2021-09-26 15:55:45
 # @FilePath: /daguan_gitee/data/code/main.sh
### 


mkdir ../prediction_result/less/
mkdir ../prediction_result/normal/
python process_data.py # 处理训练数据和测试数据
# 1.预测
## 1.1先对少样本数据预测
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
#                 --load_model_path ../user_data/fine_tune_model/less/D_replace_60_less
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