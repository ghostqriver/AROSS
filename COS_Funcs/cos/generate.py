import numpy as np
import pandas as pd

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


def Smote_Generator(area,num,*minlabel):
    '''
    A self defined SMOTE function which is given a area object, generate new points on the range(line) from the rep_point to its minority class neighbors.
    '''
    if len(minlabel) == 0:
        minlabel,_ = get_label_in_areas(area)
    else:
         minlabel = minlabel[0]

    new_points=[]
    min_neighbors = area.nearest_neighbor[area.nearest_neighbor_label==minlabel]

    for i in range(num):
        index=np.random.randint(0,len(min_neighbors))
        new_points.append(area.rep_point+np.random.rand()*(min_neighbors[index]-area.rep_point))

    return np.array(new_points)


def Gaussian_Step(area,radius_new,num,scale):
    
    new_points = []
    for i in range(num):
        ratio = np.random.normal(scale=scale,size = len(area.rep_point))
        # To ensure the ratio should in the range [-1,1]
        while (ratio>1).any() or (ratio<-1).any():
            ratio = np.random.normal(scale=scale,size = len(area.rep_point))
        new_points.append(area.rep_point + radius_new * ratio)
    return new_points


def Gaussian_Generator(area,num,*scale):

    new_points = []
    radius_new = np.sqrt(2) * area.radius /2 
    if len(scale) == 0:
        scales = [0.1,0.3,0.9]
        num0 = int(num/3)
        nums = [num0,num0,num-2*num0]
        for scale,num in zip(scales,nums):
            new_points += Gaussian_Step(area,radius_new,num,scale)
    else:
        scale = scale[0]
        new_points = Gaussian_Step(area,radius_new,num,scale)
    
    return np.array(new_points)