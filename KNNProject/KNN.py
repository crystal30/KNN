# KNN algorithm
import math
import numpy as np
from collections import Counter
def KNN(X_train,y_train,newx,k):
    '''
    :param X_train: Training data sets
    :param y_train: Lable
    :param k: k of KNN, the number of near value
    :param x: data which need to predict
    :return:predict value
    '''
    # 计算出newx 到每个点的欧式距离
    distances = np.array([math.sqrt(i)
                          for i in np.sum((X_train - newx) ** 2, axis=1)])

    nearest = distances.argsort()  # 将距离从小大大排序，取出arg
    top_y = y_train[nearest[:k]]
    votes = Counter(top_y)
    predict_value = votes.most_common(1)[0][0]

    return predict_value



