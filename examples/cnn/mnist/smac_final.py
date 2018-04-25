import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.cnn.mnist.mnist import get_data
from examples.cnn.cnn import cnn, select_gpu

select_gpu()
params = [1.39895725e+02,
          1.48324395e+02,
          4.55600341e+01,
          1.49942994e+02,
          1.83796006e+02,
          1.31789931e+01,
          7.32852875e-01,
          5.67360193e-03,
          5.33543975e-01,
          4.48722440e-01,
          6.34694890e-01,
          3.81097889e-01,
          4.63393141e-02,
          5.29944498e-02,
          1.96118275e-02,
          8.78378662e-02]

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(params, (X[train], Y[train], X[test], Y[test])))
print(np.array(ret))
