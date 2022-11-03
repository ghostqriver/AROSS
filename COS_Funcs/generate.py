import numpy as np

def MySmote(rep_point,X,num):
    '''
    A self defined SMOTE function which is given a representative point and its safe area(neighbors), generate new points on the range(line) from the rep_point to neighbors.
    '''

    
    new_points=[]
    for i in range(num):
        index=np.random.randint(0,len(X))
        new_points.append(rep_point+np.random.rand()*(X[index]-rep_point))
    return np.array(new_points)


def Random_Direction(area,num):
    '''
    A self defined function which is given a representative point and its radius,  generate new points in the area randomly
    '''
    
    new_points=[]
    for i in range(num):
        ratio = np.random.rand(len(area.rep_point))
        new_points.append(area.rep_point + area.radius * ratio)
    return np.array(new_points)