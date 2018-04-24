import numpy as np
from sklearn.model_selection import StratifiedKFold

from examples.cnn.cifar10.cifar10 import get_data
from examples.cnn.cnn import cnn, select_gpu

select_gpu()
params = [23.60354983076994, 490.14152186740233, 111.68555087043214, 129.74493665913906, 423.8734676989647,
          198.3610011114455,
          0.16961793785478674, 0.49727464470661287, 0.7059263168631477, 0.2954049199450184, 0.2403409396251418,
          0.3692255461469338, 0.03459834711189307, 0.04745673246005165, 0.06077110312934916, 0.03008818154171109]

x_train, y_train, x_test, y_test = get_data()
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))
k_fold = StratifiedKFold(n_splits=10, shuffle=False, random_state=7)
ret = []
y_stub = np.random.randint(0, 10, X.shape[0])
for train, test in k_fold.split(X, y_stub):
    ret.append(cnn(params, (X[train], Y[train], X[test], Y[test])))
print(np.array(ret))
