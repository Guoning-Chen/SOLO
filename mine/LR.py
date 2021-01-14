from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import numpy as np


def read_json(json_path):
    """
    从json文件中读取特征和标签，分别以numpy数组返回
    Returns: (ndarray, ndarray)
    """
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
    return np.asarray(features), np.asarray(labels)


def train_lr():
    # 从json文件中读取特征和标签
    np_x, np_y = read_json()
    # 展示训练数据
    con = np.concatenate((np_x, np_y.reshape((np_y.shape[0], 1))), axis=1)
    print(con)
    # 创建svm pipeline对象
    pipe = make_pipeline(StandardScaler(),  # 特征缩放（归一化）
                         LogisticRegression())
    # load the iris dataset and split it into train and test sets
    # 3类，每类50个数据，每个数据包含鸢尾花的4个属性
    # X, y = load_iris(return_X_y=True)
    # x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(np_x, np_y,
                                                        random_state=0)
    print("训练样本：%d, 测试样本：%d" % (x_train.shape[0], x_test.shape[0]))
    # 训练
    pipe.fit(x_train, y_train)
    # 测试
    predictions = pipe.predict(x_test)
    print("prediction: \n", predictions)
    confusion = np.zeros((4, 4))  # 混淆矩阵
    for pred, label in zip(predictions, [i for i in y_test]):
        confusion[pred][label] += 1
    print("confusion matrix: \n", confusion)
    test_acc = accuracy_score(predictions, y_test)
    print("test accuracy: \n", test_acc)
    # 保存模型
    model_path = 'D:/temp/LR-%d.pipe' % (int(test_acc * 10000))
    joblib.dump(pipe, model_path)


def lr_infer(lr_cp_path, json_path):
    pipe = joblib.load(lr_cp_path)
    x_test, y_test = read_json(json_path)
    predictions = pipe.predict(x_test)
    print("prediction: \n", predictions)
    confusion = np.zeros((4, 4))  # 混淆矩阵
    for pred, label in zip(predictions, [i for i in y_test]):
        confusion[pred][label] += 1
    print("confusion matrix: \n", confusion)
    test_acc = accuracy_score(predictions, y_test)
    print("test accuracy: \n", test_acc)

    return test_acc


if __name__ == "__main__":
    train_lr()