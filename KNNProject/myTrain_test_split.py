import numpy as np

def myTrain_test_split(X,y,test_ratio=0.2,seed = None):
    '''

    :param X: input data set
    :param y: input lable set
    :param test_ratio: the proportion of test data set
    :param seed: random seed
    :return: X_train,y_train,X_test,y_test
    '''

    assert X.shape[0] == y.shape[0], \
        "the len of the X must be equal to the len the y"
    assert 0 <= test_ratio <= 1, \
        "test_ratio must be more than 0 and less than 1"
    if seed:
        np.random.seed(seed)

    # 打乱数据
    shuffle_indexes = np.random.permutation(len(X))
    test_number = int(len(X) * test_ratio)
    X_test = X[shuffle_indexes[:test_number], :]
    y_test = y[shuffle_indexes[:test_number]]
    X_train = X[shuffle_indexes[test_number:], :]
    y_train = y[shuffle_indexes[test_number:]]

    return X_train,X_test,y_train,y_test