import numpy as np
import pandas as pd
from COS_Funcs.cos.nearest_neighbor import nn_kd

def get_label_in_areas(area):
    
    labels = area.nearest_neighbor_label

    valuecounts = pd.Series(labels).value_counts().index
    minlabel = valuecounts[0]
    if len(valuecounts) > 1:
        majlabel = valuecounts[1]
    else:
        majlabel = None

    return minlabel,majlabel

def check_parameter(function,minlabel,Gaussian_scale):
    '''
    Return the suitable parameter for the generating functions
    '''

    if function == Smote_Generator:
        return minlabel
    
    elif function == Gaussian_Generator:
        return Gaussian_scale


def Smote_Generator(area,num,**minlabel):
    '''
    A self defined SMOTE function which is given a area object, generate new points on the range(line) from the rep_point to its minority class neighbors.
    '''
    # if len(minlabel) == 0:
    minlabel,_ = get_label_in_areas(area)
    # else:
    #      minlabel = minlabel[0]

    new_points=[]
    min_neighbors = area.nearest_neighbor[area.nearest_neighbor_label==minlabel]

    for i in range(num):
        index=np.random.randint(0,len(min_neighbors))
        new_points.append(area.rep_point+np.random.rand()*(min_neighbors[index]-area.rep_point))

    return np.array(new_points)

def filter(new_point,tree,k=5,y_train=None,min_label=1):
    inds,dists = nn_kd([new_point],k,tree)
    # if len(inds)<k:
    #     print('!!!!!!!!!!!!!!!!')
    return not all(y_train[inds]==min_label)

    
def Gaussian_Generator_(area,radius_new,num,scale,tree=None,y_train=None,min_label=1):
       
    if area.num_min < 5:
        k = area.num_min
    else:
        k = 5
         
    new_points = []
    filter_cnt = 0
    while len(new_points) < num:
        ratio = np.random.normal(scale=scale,size = len(area.rep_point))
        
        # To ensure the ratio should in the range [-1,1]
        while (ratio>1).any() or (ratio<-1).any():
            ratio = np.random.normal(scale=scale,size = len(area.rep_point))
        new_point = area.rep_point + radius_new * ratio
        
        if tree is None:
            # safe area
            new_points.append(new_point)
        else:
            # half safe area
            
            if not filter(new_point,tree,k,y_train=y_train,min_label=min_label):
                new_points.append(new_point)
            else:
                # Record when filtering
                filter_cnt += 1 
                if filter_cnt == 50:
                # When the sample is filtered many times, the minority area in half safe area might be very small or out of square
                # Then set the radius to orginal radius
                    radius_new = (radius_new * 2) / np.sqrt(2)
                if filter_cnt > 1000 and len(new_points)==0:
                # When this area is filtered by too many times without any sample generated in, then break the loop
                    break    
    return new_points

def Gaussian_Generator(area,num,scale=None,tree=None,y_train=None,min_label=1):
    
    # if tree is not None:
    #     # To avoid running too long time in some rare case that minority instances located at four corners
    #     # Samples will be generated out of circle, but it is still safe due to filter
    #     radius_new = area.radius
    # else:
    radius_new = np.sqrt(2) * area.radius /2 
    
    if scale is None:
        scale = 0.8

    new_points = Gaussian_Generator_(area,radius_new,num,scale,tree,y_train,min_label)
    
    return np.array(new_points)