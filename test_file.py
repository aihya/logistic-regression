from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SLR
from logistic_regression import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


v = load_digits()
   
def unpickle(file):
    with open(file, 'rb') as fo:
        _dict = pickle.load(fo, encoding='bytes')
    return _dict

X_train = None
y_train = None
#X_train = np.zeros([50000, 32, 32], dtype=int)

def convToImg(imgArr):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(imgArr.reshape(3, -1).T.reshape(32, 32, 3), rgb_weights)

for i in range(1, 2):
    i_dict = unpickle("/Users/aihya/goinfre/cifar-10-batches-py/data_batch_{}".format(str(i)))
    if X_train is None:
        X_train = i_dict[b'data']
    else:
        X_train = np.concatenate((X_train, i_dict[b'data']), axis=0)
    if y_train is None:
        y_train = i_dict[b'labels']
    else:
        y_train = np.concatenate((y_train, i_dict[b'labels']))
"""
print("Converting...")
for idx, imgArr in enumerate(_X_train):
    print(idx)
    X_train[idx] = convToImg(imgArr)

print("Done converting.")
"""
#X_train, X_test, y_train, y_test = train_test_split(v.data, v.target,
#        test_size=0.2)

#X_train = np.array(X_train, dtype=np.float128)
lr = LR(normalize=False)
slr = SLR(multi_class='ovr')

