import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.cnn.cifar10.cifar10 import get_data
from examples.cnn.cnn import cnn, select_gpu

"""
Best result: 0.238800
Job-id: 20005
Parameters: 
name: "width1"
int_val: 110
name: "width2"
int_val: 18
name: "width3"
int_val: 87
name: "width4"
int_val: 53
name: "width5"
int_val: 97
name: "width6"
int_val: 275
name: "dropout1"
dbl_val: 0.05
name: "dropout2"
dbl_val: 0.446227673193
name: "dropout3"
dbl_val: 0.149082967174
name: "dropout4"
dbl_val: 0.247763545073
name: "dropout5"
dbl_val: 0.0716215130478
name: "dropout6"
dbl_val: 0.05
name: "regularizer1"
dbl_val: 0.0001
name: "regularizer2"
dbl_val: 0.000145183924707
name: "regularizer3"
dbl_val: 0.00139104840186
name: "regularizer4"
dbl_val: 0.00900573513475
"""
select_gpu()
params = [110, 18, 87, 53, 97, 275,
          0.05, 0.446227673193, 0.149082967174, 0.247763545073, 0.0716215130478, 0.05,
          0.05, 0.0001, 0.000145183924707, 0.00139104840186, 0.00900573513475]

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(params, (X[train], Y[train], X[test], Y[test])))
print(np.array(ret))
