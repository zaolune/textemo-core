from typing import List,Union
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
from sklearnex import patch_sklearn
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

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


def forest(x_train,y_train)->Union[RandomForestClassifier,GridSearchCV]:
     # 随机森林
    estimator = RandomForestClassifier(n_estimators=1200)
    # param = {"n_estimators": [1050,1200,1300,1400]}
    # estimator = GridSearchCV(estimator, param_grid=param, cv=2)
    estimator.fit(x_train,y_train)
    joblib.dump(estimator,'./model/forest.pkl')
    # estimator = joblib.load('./model/forest.pkl')
    return estimator





def main():
    data = []
    data += read_file('./data/ChnSentiCorp_htl_all.txt')
    print("数据集总数：",len(data))
    stop_words = read_stop_words('./cn_stopwords.txt')
    dataset = {
        "data":[],
        "target":[]
    }
    for item in data:
        dataset['data'].append(cut_words(item[1]))
        dataset['target'].append(item[0])
    
    x_train, x_test, y_train, y_test = train_test_split(dataset['data'],dataset['target'], random_state=1)
    print("训练集数据集总数：",len(x_train))
    print("测试集数据集总数：",len(x_test))
    
    transfer = TfidfVectorizer(stop_words=stop_words)
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 保存特征提取模型
    read_or_write_vectorizer('./vectorizer/TfidfVectorizer.pkl','w',transfer)
    
    estimator = forest(x_train,y_train)

    y_pred = estimator.predict(x_test)
    print(classification_report(y_test,y_pred))
    print("测试：",estimator.score(x_test,y_test))
    # print("在交叉验证当中验证的最好结果：", estimator.best_score_)
    # print("gc选择了的模型K值是：", estimator.best_estimator_)
    text = "体验很好"
    feature = cut_words(text)
    print("分词后的feature=",feature)
    feature = transfer.transform([feature])
    print("转换之后的feature=",feature)
    print(estimator.predict(feature))
    
def use():
    cv = read_or_write_vectorizer('./vectorizer/TfidfVectorizer.pkl','r')
    estimator = joblib.load('./model/forest.pkl')
    text = "体验很好"
    feature = cut_words(text)
    print("分词后的feature=",feature)
    feature = cv.transform([feature])
    print("转换之后的feature=",feature)
    print(estimator.predict(feature))

# main()
use()