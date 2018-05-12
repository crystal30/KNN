import numpy as np
from sklearn import datasets
from KNNClf import KNNClf
iris = datasets.load_iris()
iris.keys()

Row_ind = np.arange(0,100,10)  #array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]）
#为简单起见,我们从第一类和第二类鸢尾花中各取出5个数据
X_train = iris.data[Row_ind,2:]
y_train = iris.target[Row_ind]

myknn_clf = KNNClf(k=7)
print(X_train.shape[0] == y_train.shape[0])
myknn_clf.fit(X_train,y_train)
