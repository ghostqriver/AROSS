'''
@brief generators 
'''

import numpy as np
import pandas as pd

from .nearest_neighbor import nn_kd


def Gaussian_Generator(area,num,tree=None,y_train=None,min_label=None):
    
    # Shrink the radiu of area / truncate1
    radius_new = np.sqrt(2) * area.radius /2 

    new_points = Gaussian_Generator_(area,radius_new,num,tree,y_train,min_label)
    
    return np.array(new_points)

def Gaussian_Generator_(area,radius_new,num,tree=None,y_train=None,min_label=None):
       
    if area.num_min < 5:
        k = area.num_min
    else:
        k = 5
        
    scale = 0.8
    size = len(area.rep_point)
    new_points = []
    filter_cnt = 0
    
    while len(new_points) < num:
        ratio = np.random.normal(scale=scale,size=size)
        
        # To ensure the ratio should in the range [-1,1] / truncate2
        while (ratio>1).any() or (ratio<-1).any():
            ratio = np.random.normal(scale=scale,size=size)
        new_point = area.rep_point + radius_new * ratio
        
        if tree is None:
            # safe area
            new_points.append(new_point)
        else:
            # half safe area
            if not filter(new_point,tree,k,y_train=y_train,min_label=min_label):
                new_points.append(new_point)
            else:
                # Avoid infinite loop
                filter_cnt += 1 
                if filter_cnt == 50:
                    # Set to orginal radius
                    radius_new = (radius_new * 2) / np.sqrt(2)
                if filter_cnt > 1000 and len(new_points)==0:
                    # Pass this area
                    break    
    return new_points


def filter(new_point,tree,k=5,y_train=None,min_label=1):
    inds,dists = nn_kd([new_point],k,tree)
    return not all(y_train[inds]==min_label)


def get_label_in_areas(area):
    
    labels = area.nearest_neighbor_label

    valuecounts = pd.Series(labels).value_counts().index
    minlabel = valuecounts[0]
    if len(valuecounts) > 1:
        majlabel = valuecounts[1]
    else:
        majlabel = None

    return minlabel,majlabel


def Smote_Generator(area,num,**minlabel):
    '''
    @brief A self defined SMOTE function for oversampling in the area
    '''
    minlabel,_ = get_label_in_areas(area)
    # else:
    #      minlabel = minlabel[0]

    new_points=[]
    min_neighbors = area.nearest_neighbor[area.nearest_neighbor_label==minlabel]

    for i in range(num):
        index=np.random.randint(0,len(min_neighbors))
        new_points.append(area.rep_point+np.random.rand()*(min_neighbors[index]-area.rep_point))

    return np.array(new_points)

