# 基于LCE算法的文本情感分类系统

LCE： [Local Cascade Ensemble]('https://lce.readthedocs.io/en/latest/generated/lce.LCEClassifier.html')

数据集：[携程网评论数据集]('https://github.com/Embedding/ChineseNlpCorpus/blob/master/datasets/ChnSentiCorp_htl_all/intro.ipynb')

基于sklearn的文本情感分析系统，1代表积极情感，0代表消极情感。
data：存放使用的数据集

model：程序运行后的模型

vectorizer：程序运行后的转换器
## Install
推荐使用虚拟环境安装依赖：
```bash
pip install -r requirements.txt
```
## Use

```bash
python lce_pca.py
```
estimator.py：随机森林

xgboost_estimator：xgboost