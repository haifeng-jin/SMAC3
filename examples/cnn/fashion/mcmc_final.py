import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.cnn.fashion.fashion import get_data
from examples.cnn.cnn import cnn, select_gpu

"""
Best result: 0.015133
Job-id: 0
Parameters: 
name: "width1"
int_val: 10
name: "width2"
int_val: 10
name: "width3"
int_val: 10
name: "width4"
int_val: 10
name: "width5"
int_val: 10
name: "width6"
int_val: 10
name: "dropout1"
dbl_val: 0.05
name: "dropout2"
dbl_val: 0.05
name: "dropout3"
dbl_val: 0.05
name: "dropout4"
dbl_val: 0.05
name: "dropout5"
dbl_val: 0.05
name: "dropout6"
dbl_val: 0.05
name: "regularizer1"
dbl_val: 0.0001
name: "regularizer2"
dbl_val: 0.0001
name: "regularizer3"
dbl_val: 0.0001
name: "regularizer4"
dbl_val: 0.0001
"""

select_gpu()
params = [10, 10, 10, 10, 10, 10,
          0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
          0.0001, 0.0001, 0.0001, 0.0001]

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(params, (X[train], Y[train], X[test], Y[test])))
print(np.array(ret))
