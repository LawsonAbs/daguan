<!--
 * @Author: LawsonAbs
 * @Date: 2021-09-04 22:07:40
 * @LastEditTime: 2021-09-22 11:03:59
 * @FilePath: /daguan/README.md
-->
# 方法
- 使用 train + test 进行预训练，而不是使用 unlabel 的数据预训练。
如果单纯使用unlabel 预训练，效果大概只有0.52 左右，使用train+test 效果可到0.592
- 可以同时结合ngram mask 进行训练，理论上可以效果会有1个点提升，但是由于时间问题，笔者未能按时跑出。



# 预训练

数据集：train.txt + test.txt 

```sh
python risk_data_grand/pretrain/pretrain.py
```

# fine-tuning(训练模型)

```sh
python risk_data_grand/train.py 
```

# predict

```sh
python run_predictor_local.py
```