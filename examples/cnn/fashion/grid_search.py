from examples.cnn.fashion.fashion import get_data
from examples.cnn.cnn import cnn

for width in [64, 128, 256]:
    for dropout in [0.25, 0.5]:
        for regularizer in [0.01, 0.001]:
            params = [width * 1.0] * 6 + [dropout] * 6 + [regularizer] * 4
            print(params)
            cnn(params, get_data())
