import math
import numpy as np
from collections import Counter

class KNNClf():

    def __init__(self, k):
        assert k>=1,"k must be valid"
        self.k = k
        self.__X_train = None
        self.__y_train = None

    def fit(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of the X_train must be equal to the size the y_trian"
        assert self.k<=X_train.shape[0],\
            "the size of the X_train must be at least k"
        self.__X_train = X_train
        self.__y_train = y_train

        return self

    def predict(self,newX):
        assert newX.ndim == 2,\
            "predict date(newX) must be 2dim"
        y_predict = [self.__predict(x) for x in newX]
        return np.array(y_predict)

    def __predict(self,x):
        # 计算出newx 到每个点的欧式距离
        distances = np.array([math.sqrt(i)
                              for i in np.sum((self.__X_train - x) ** 2, axis=1)])

        nearest = distances.argsort()  # 将距离从小大大排序，取出arg
        top_y = self.__y_train[nearest[:self.k]]
        votes = Counter(top_y)
        predict_value = votes.most_common(1)[0][0]

        return predict_value

