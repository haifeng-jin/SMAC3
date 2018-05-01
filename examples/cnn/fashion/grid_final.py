import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.cnn.fashion.fashion import get_data
from examples.cnn.cnn import cnn, select_gpu

select_gpu()
params = [128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.01, 0.01, 0.01]

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(params, (X[train], Y[train], X[test], Y[test])))
print(np.array(ret))
