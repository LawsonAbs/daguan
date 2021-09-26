<!--
 * @Author: LawsonAbs
 * @Date: 2021-09-25 16:43:25
 * @LastEditTime: 2021-09-26 15:07:23
 * @FilePath: /daguan_gitee/data/code/submission/readme.md
-->
由于团队疏忽，未能保存所有模型，导致部分数据无法重现，其包括如下文件：
- submission_0905_0.583.csv ： 一个较好的提交结果，于2021/09/05 23:40 提交A榜的成绩
- submission_balance_10_num_30.csv ：用于解决小样本问题。对于每个类别的数据只随机抽取30个样本，训练 10 epoch，得到的结果。
- submission_balance_10_num_60.csv ：用于解决小样本问题。对于每个类别的数据值随机抽取60个样本，训练 10 epoch，得到的结果。

本团队给出如下两种方案解决：
# 方案一
如果需要完全复现团队B榜提交结果，还请按照run.sh脚本将文件 data/code/submission_0905_0.583.csv 移动到 data/prediction_result/normal 下，将 data/code/submission_balance_10_num_30.csv 和 data/code/submission_balance_10_num_60.csv 移动到 prediction_result/less 下，这样可**保证完全复现**提交结果。

# 方案二
考虑到贵司若坚持需要使用完全的预测模型重跑，则可以按照我在 run.sh 中的脚本中的说明进行相应操作。使用D_0.583_replace 模型生成近似于
submission_0905_0.583.csv 的结果； 使用D_60_replace,D_90_replace 来代替生成 submission_balance_10_num_30.csv 和 submission_balance_10_num_60.csv。 
- 由于数据是随机抽取的，所以**几乎不能完全复现这两个文件的内容**
- 但是**存在比B榜提交结果好的可能**（因为训练得到的D_60_replace，以及 D_90_replace 模型使用了更好好的预训练模型微调得到）。