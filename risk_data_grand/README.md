目前的结果
1. 尝试train_set+test_set nezha-fgm fgm 0.52
2. 尝试train_set+test_set+2w_unlabel_data nezha-fgm fgm+sparemax 0.54
3. 尝试train_set+test_set+2w_unlabel_data nezha-fgm fgm+labelssmothing+R-dropout 0.515
4. 尝试train_set+test_set+2w_unlabel_data nezha-fgm fgm+labelssmothing 


# 预训练
python pretrain_code/pretrain.py --model_name="nezha-cn-base" \
                                --train_fgm=True \
                                --model_save_path="nezha-base-fgm" \
                                --model_type="nezha" \
                                --batch_size=8 \
                                --gradient_accumulation_steps=2 \
                                --num_epochs=150 \
                                --fgm_epsilon=1.0 \
                                --manual_seed=124525601

目前的预训练模型是用训练数据和测试集还有2w的无标签数据训练得到的


# fine-tuning(训练模型)
python run_classify_local.py 

# predict
python run_predictor_local.py


# 其它
pretrain_test.txt 是由test.csv转换得到，去掉了id，仅保留了文本。
pretrain_train.txt 是由train.csv转换得到，去掉了id和标签，仅保留了文本。