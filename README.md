<!--
 * @Author: LawsonAbs
 * @Date: 2021-09-04 22:07:40
 * @LastEditTime: 2021-09-22 15:31:39
 * @FilePath: /daguan/README.md
-->
# 1. 算法思想
针对本赛题，LModel队伍将其视作一种**分类任务**，采取的方法是：使用基于**预训练+微调** 的方法来解决本任务。

# 2 任务分析
## 2.1 标签映射
将每个标签映射到一个独立的id，如将'5-24'映射到0，这样便构成了一个标签到id的唯一映射。这样就将任务转换成一个分类任务。

## 2.2 数据分析
数据分析主要围绕“文本长度”，“类别数据”进行分析。
针对文本长度，主要分析 `train.txt+test.txt` (+表示拼接两个文本)； `datagrand_2021_unlabeled_data` （下称无标签数据）中 `title` 字段数据长度，二者长度分布如下：
![title_analysis](train+test.png)
![title_analysis](title.png)
可以很明显看到，二者在文本长度这一特征上有明显的不同，所以让我们选择以无标签数据简单预训练，以 `train.txt + test.txt` 为主要预训练。

## 2.3 模型分析
处理分类任务的方法有很多，机器学习方法有：KNN算法；朴素贝叶斯；SVM等等。深度学习方法有：使用预训练模型+softmax分类。考虑到深度学习在分类问题上的优越表现，我们选择使用深度学习方法作为本问题的解决方案。在深度学习中我们使用的是 `chinese-roberta-wwm-ext-large` 。

## 2.4 模型融合
考虑到单模型在一个问题上可能存在短板，我们选择使用四个比较健壮的模型来融合得到最后的结果，即submission_ensemble.csv；同时考虑到小类别数据在上述四个模型上的效果不佳，我们采取重采样+数据增强的方法针对少数类别的样本进行一个单独的训练，将得到的模型进行一个融合，得到融合结果submission_less_ensemble.csv。在小类别数据上，使用submission_less_ensembel.csv覆盖上述文件 submission_ensemble.csv 得到最后的提交文件 submission_best_combine.csv。


# 3. 详细实现
## 3.1 预训练
- step1. 使用无标签数据进行一个简单的预训练操作，训练1w step，得到模型checkpoint_1w
- step2. 在上述模型的基础上使用 `train.txt+test.txt`数据进行一个 90000 step 的预训练，得到预训练的最终模型 checkpoint_1w_9w

本部分使用的参数详见代码 ./risk_data_grand/pretrain/pretrain.py 中的参数。

## 3.2 微调
- step1. 使用预训练好的模型checkpoint_1w_9w 使用 `train.txt` 数据集进行微调。微调取 `epoch=10` 作为最后的微调结果模型，即 checkpoint_1w_9w_epoch_10。


## 3.3 预测

使用上面预训练得到的模型进行预测，预测结果写入到 submission.csv 中。

# 4. 代码结构
代码结构如下：
```c
|-- data
    |-- user_data
	    |-- 参赛者模型文件
	    |-- 其他文件等
    |-- prediction_result
	    |-- result.csv
    |-- code
	    |-- 这里是docker执行时需要的代码
    |-- raw_data
        |-- 比赛的数据集文件（这里执行时会被替换为官方网站上的数据）
    |--README.md
```