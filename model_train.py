# -*- coding: utf-8 -*-
# model_train.py
# Python 3.6.8, TensorFlow 2.3.0, Keras 2.4.3
# 导入模块
import keras as K
import pandas as pd
from sklearn.model_selection import KFold


# 读取CSV数据集
# 该函数的传入参数为csv_file_path: csv文件路径
def load_data(sv_file_path):
    iris = pd.read_csv(sv_file_path)
    target_var = 'class'  # 目标变量
    # 数据集的特征
    features = list(iris.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = iris[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target, 将目标变量转化为类别变量
    iris['target'] = iris[target_var].apply(lambda x: Class_dict[x])

    return features, 'target', iris


# 创建模型
def create_model():
    init = K.initializers.glorot_uniform(seed=1)
    simple_adam = K.optimizers.Adam()
    model = K.models.Sequential()
    model.add(K.layers.Dense(units=5, input_dim=4, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=6, kernel_initializer=init, activation='relu'))
    model.add(K.layers.Dense(units=3, kernel_initializer=init, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
    return model


def main():
    # 1. 读取CSV数据集
    print("Loading Iris data into memory")
    n_split = 10
    features, target, data = load_data("./iris_data.csv")
    x = data[features]
    y = data[target]
    avg_accuracy = 0
    for train_index, test_index in KFold(n_split).split(x):
        print("test index: ", test_index)
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print("create model and train model")
        model = create_model()
        model.fit(x_train, y_train, batch_size=1, epochs=80, verbose=0)

        print('Model evaluation: ', model.evaluate(x_test, y_test))
        avg_accuracy += model.evaluate(x_test, y_test)[1]

    print("K fold average accuracy: {}".format(avg_accuracy/n_split))


main()