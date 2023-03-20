from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import random
import numpy as np



def standard(x1,center=None,outlier1=None,outlier2=None):
    ss = StandardScaler()
    ss.fit(x1)
    X_ss = ss.transform(x1)
    if outlier1 and outlier2 and center:
        center_ss = ss.transform([center])[0]
        outlier1_ss = ss.transform([outlier1])[0]
        outlier2_ss = ss.transform([outlier2])[0]
        return  X_ss,center_ss,outlier1_ss,outlier2_ss
    else:
        return  X_ss
    
def makeData(datanum):
    '''
    @Making the positive correlation data
    '''
    x1,_,center = make_blobs(n_samples=datanum, n_features=2, centers=1, random_state=random.randint(0, 1000),center_box=(0, 10.0),return_centers=True)
    transformation = [[0.558, 23], [0.678, 11564]]
    x1 = np.dot(x1, transformation)
    center = np.dot(center, transformation)[0]
    return x1,center