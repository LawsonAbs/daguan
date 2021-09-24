###
 # @Author: LawsonAbs
 # @Date: 2021-09-24 11:03:42
 # @LastEditTime: 2021-09-24 20:30:27
 # @FilePath: /daguan_gitee/code/pretrain.sh
### 

echo ">>>>>>>>>>>> 开始使用模型 bert-base-chinese 进行预训练。 <<<<<<<<<\n\n"
cd pretrain
#============================= step 1. 使用 bert-base-chines 预训练 =============================
python pretrain.py --vocab_file ../../user_data/data/vocab_1.txt  \
                   --train_file_path ../../user_data/data/all.txt  \
                   --model_name   bert-base-chinese \
                   --num_epoch 200
echo ">>>>>>>>>>>> 在bert-base-chinese基础上的预训练结束！<<<<<<<<<\n\n"

#============================= step 2. 使用 chinese-roberta-wwm-ext-large 预训练 =============================
# 以vocab_2.txt 在large 上训练
# 先在 2.4G 的数据上预训练
echo ">>>>>>>>>>>> 在chinese-roberta-wwm-ext-large 基础上进行预训练 <<<<<<<<<\n\n"
python pretrain.py --vocab_file ../../user_data/data/vocab_2.txt  \
                   --train_file_path ../../user_data/data/1.txt  \
                   --model_name chinese-roberta-wwm-ext-large  \
                   --num_epoch 100

# 再在 all.txt（train.txt+test.txt） 上预训练，训练200 epoch
python pretrain.py --vocab_file ../../user_data/data/vocab_2.txt  \
                   --train_file_path ../../user_data/data/all.txt  \
                   --model_name   ../../user_data/pretrain_model/checkpoint_epoch_100_large \
                   --num_epoch 200
echo ">>>>>>>>>>>> 在 chinese-roberta-wwm-ext-large 基础上的预训练结束！<<<<<<<<<\n\n"