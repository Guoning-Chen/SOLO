from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import numpy as np


if __name__ == "__main__":
    # 读取特征
    json_path = 'D:/temp/results/train340.json'
    state = json.load(open(json_path))
    features = []
    labels = []
    for img_info in state['content']:
        chain_num = 0
        chain_density = 0.0
        img_path = img_info['img']
        label = int(img_info['label'])
        labels.append(label)
        for m in img_info['mask']:
            if (m['label'] == 'chain'):
                chain_num += 1
                chain_density += m['ratio']
        features.append((chain_num, chain_density))
    np_x, np_y = np.asarray(features), np.asarray(labels)
    con = np.concatenate((np_x, np_y.reshape((np_y.shape[0], 1))), axis=1)
    print(con)
    # 创建pipeline对象
    pipe = make_pipeline(StandardScaler(),   # 特征缩放（归一化）
                         LogisticRegression())
    # load the iris dataset and split it into train and test sets
    # 3类，每类50个数据，每个数据包含鸢尾花的4个属性
    # X, y = load_iris(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(np_x, np_y, random_state=0)
    print("训练样本：%d, 测试样本：%d" % (X_train.shape[0], X_test.shape[0]))
    # 训练
    pipe.fit(X_train, y_train)
    # 测试
    predictions = pipe.predict(X_test)
    print("prediction: \n", predictions)
    confusion = np.zeros((4, 4))  # 混淆矩阵
    for pred, label in zip(predictions, [i for i in y_test]):
        confusion[pred][label] += 1
    print("confusion matrix: \n", confusion)
    print("test accuracy: \n", accuracy_score(predictions, y_test))

