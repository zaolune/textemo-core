from typing import List,Union
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from lce import LCEClassifier
import joblib
import pickle
from sklearnex import patch_sklearn
from sklearn.metrics import classification_report
patch_sklearn()

jieba.load_userdict('./dictionary.txt')
def read_file(file:str)->List:
    """
    从文件中读取数据
    """
    data = []
    label_end = 1
    sentence_begin = 5

    with open(file, 'r',encoding='utf-8')as f:
        for line in f:
            label = line[0:label_end]
            sentence = line[sentence_begin:].strip()
            data.append([label,sentence])
        return data


def read_stop_words(file)->List:
    """
    从文件中读取stopwords
    """
    stop_words = []
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.strip())
        return stop_words

def cut_words(sentence:str)->List:
    return " ".join(list(jieba.cut(sentence)))

def read_or_write_vectorizer(file,type,transfer:TfidfVectorizer=None):
    """
    读取或保存特征提取模型
    """
    if type == 'r':
        return  pickle.load(open(file,'rb'))
    if type == 'w':
        with open(file,'wb') as f:
            pickle.dump(transfer,f)


def lce(x_train,y_train)->LCEClassifier:
    estimator = LCEClassifier(n_jobs=2,random_state=0)
    estimator.fit(x_train.toarray(),y_train)
    joblib.dump(estimator,'./model/lce.pkl')
    return estimator

    

def main():
    data = []
    # data += read_file('./data/disabled.txt')
    data+=read_file('./data/ChnSentiCorp_htl_all.txt')
    print("数据集总数：",len(data))
    stop_words = read_stop_words('./cn_stopwords.txt')
    dataset = {
        "data":[],
        "target":[]
    }
    for item in data:
        dataset['data'].append(cut_words(item[1]))
        dataset['target'].append(item[0])
    
    x_train, x_test, y_train, y_test = train_test_split(dataset['data'],dataset['target'], random_state=1,train_size=2000)
    print("训练集数据集总数：",len(x_train))
    print("测试集数据集总数：",len(x_test))
   
    transfer = TfidfVectorizer(stop_words=stop_words)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 保存特征提取模型
    read_or_write_vectorizer('./vectorizer/TfidfVectorizer.pkl','w',transfer)
  
    estimator = lce(x_train,y_train)

    y_pred = estimator.predict(x_test.toarray())
    print(classification_report(y_test,y_pred))
    print("测试：",estimator.score(x_test.toarray(),y_test))
    text = "体验很好"
    feature = cut_words(text)
    print("分词后的feature=",feature)
    feature = transfer.transform([feature])
    print("转换之后的feature=",feature)
    print(estimator.predict(feature.toarray()))
 
main()

def use():
    transfer = read_or_write_vectorizer('./vectorizer/TfidfVectorizer.pkl','r')
    estimator = joblib.load("./model/lce.pkl")
    text = "环境太糟糕了，建议不要来"
    feature = cut_words(text)
    print("分词后的feature=",feature)
    feature = transfer.transform([feature])
    print("转换之后的feature=",feature)
    print(estimator.predict(feature.toarray()))

# use()