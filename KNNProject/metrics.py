import numpy as np

def accaracy_score(y_predict,y_test):
    '''

    :param y_predict: the predict lable of test data set
    :param y_test: the true lable of the test data set
    :return: accaracy
    '''
    assert y_predict.shape[0] == y_test.shape[0],\
        "the len of the y_predict must be equal to the len of the y_test "
    return sum(y_predict == y_test)/len(y_test)
