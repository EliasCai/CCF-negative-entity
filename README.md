# 金融信息负面及主体判定
使用pytorch+bert的方式，将问题转换成多标签的预测问题，[比赛地址](https://www.datafountain.cn/competitions/353 )

### 训练
CUDA_VISIBLE_DEVICES=0 python code/train.py

### 预测
CUDA_VISIBLE_DEVICES=0 python code/infer.py

### ToDo
1. 数据增强，将实体进行替换后训练
2. 数据匹配，有接近10%的实体不能在TEXT中完全匹配，需要做一个近似匹配

### 线上分数
0.88