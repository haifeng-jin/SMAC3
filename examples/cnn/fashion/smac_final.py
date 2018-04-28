import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.cnn.fashion.fashion import get_data
from examples.cnn.cnn import cnn, select_gpu

select_gpu()
params = [2.99238479e+02,
          3.82071856e+02,
          3.27443621e+02,
          4.38605543e+02,
          4.95966799e+02,
          4.66952157e+02,
          5.63115433e-01,
          5.54686304e-01,
          3.45478902e-01,
          9.75819800e-01,
          9.57088605e-01,
          1.66480712e-01,
          9.97939190e-02,
          3.30969204e-02,
          9.21902344e-02,
          5.04329819e-02]

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(params, (X[train], Y[train], X[test], Y[test])))
print(np.array(ret))
