from sklearn.datasets import make_blobs
import random
import numpy as np

def makeData(datanum):
    x1,_,center = make_blobs(n_samples=datanum, n_features=2, centers=1, random_state=random.randint(0, 1000),center_box=(0, 10.0),return_centers=True)
    transformation = [[0.558, 23], [0.678, 11564]]
    x1 = np.dot(x1, transformation)
    center = np.dot(center, transformation)[0]
    return x1,center