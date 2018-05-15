import numpy as np

class StandardScaler():

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self,X):
        '''

        :param X: X usually is train data(X_train)
        :return: self
        '''
        assert X.ndim == 2, "We must input a matrix"
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self

    def transform(self,X):
        '''
        self.mean_ self.scale_ is none
        len(self.mean_) == X.shape[1]
        can add assert
        :param X: X need to normalize
        :return:
        '''
        assert X.ndim == 2, "the dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None,\
            "must fit before transform"

        standX = np.empty(shape=X.shape, dtype= float)
        for i in range(len(self.mean_)):
            standX[:,i] = (X[:,i]-self.mean_[i])/self.scale_[i]

        return standX


