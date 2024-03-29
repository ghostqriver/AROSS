'''
@brief COS's brute force implementation
@details Slow but works as a ground truth when developing speeding up versions
@author yizhi
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools

from utils import get_labels
from utils.dist import calc_cov_i,calc_dist
from cluster import cure
from utils import visualize as V
from . import generate as G

class Area():
    '''
    The object of areas around the representative points
    Attributes:
        .rep_point: the representative point which the area generated from 
        .nearest_neighbor: the neighbors in its safe area
        .nearest_neighbor_index: the neighbors' index from the original dataset
        .nearest_neighbor_label: the neighbors' label from the original dataset
        .num_neighbor: number of neighbors in this area
        .radius: the radius of this area (the representative point to its furthest neighbor)
        .safe: the safe arrtibute of this area, 1:min_all_safe 2:min_half_safe 3:max_all_safe 4:max_half_safe
    '''
    def __init__(self,rep_point) :
        self.rep_point = rep_point
        self.nearest_neighbor = []
        self.nearest_neighbor_index = []
        self.nearest_neighbor_label = []
        self.num_neighbor = 0
        self.radius = 0
        self.safe = 0 # 1 min_all_safe 2 min_half_safe 3 max_all_safe 4 max_half_safe

    def renew_neighbor(self,data,index,label):
        '''
        Append the neighbor info to lists
        Will be used when expand the all safe area
        '''
        self.nearest_neighbor.append(data)
        self.nearest_neighbor_index.append(index)
        self.nearest_neighbor_label.append(label)


    def neighbor_toarray(self):
        '''
        Would be call when finish generating the safe area, for convert the lists to array
        '''
        self.nearest_neighbor = np.array(self.nearest_neighbor)
        self.nearest_neighbor_index = np.array(self.nearest_neighbor_index)
        self.nearest_neighbor_label = np.array(self.nearest_neighbor_label)


    def del_neighbor(self):
        '''
        Remove the final neighbor
        Will be used when shrink the half safe area
        '''
        del self.nearest_neighbor[-1]
        del self.nearest_neighbor_index[-1]
        del self.nearest_neighbor_label[-1]

    def renew_paras(self,safe):
        '''
        Will be called after generate the safe area
        '''
        self.num_neighbor = len(self.nearest_neighbor)
        self.safe = safe 
        self.radius = calc_dist(self.rep_point,self.nearest_neighbor[-1])

    def gen_safe_area(self,X,y,minlabel=None,majlabel=None,k=3,shrink_half=False,expand_half=False):
        '''
        Generate all the area object (representative points) 's area:
        if k neighbors of rep_point are all belonging to the minority class --> min safe area
        if more than k/2 neighbors of rep_point are belonging to the minority class --> min half safe area
        X: data
        y: label
        minlabel,majlabel: given the label of minority class and majority class, if None will be set from the dataset automatically (only work in binary classification case)
        k: should be an odd value, it will check k nearest neighbor of the representative points to decide safe or half safe, default 3
        shrink_half: if true it will try to shrink the half safe area to exclude the furthest majority class's point out of its neighbor until there is no change, default false, after shrink if satisfied all safe condition, then it will set that half safe area to all safe area 
        expand_half: if true it will try to expand the half safe area to contain more the nearest minority class's point into its neighbor until there is no chang, default false 
        If the shrink_half and expand_half are both True, it will only do the shrinking
        '''
        safe = 0

        if minlabel == None and majlabel ==None:
            minlabel,majlabel = get_labels(y)
        # Calculate the distances
        tmp_dist = []
        for data in X:
            tmp_dist.append(calc_dist(self.rep_point,data))

        count = 0 

        # Get k neighbors  
        '''
        Mark:
        If we use filter to check whether a new generated point is good enough or not, we can set k=1
        then there will be more all safe areas, but due to the logic of this part code, there will never
        be a half safe area, so if set k=1 this part need to be considered.
        '''    
        while(count < k): 
            index = np.argmin(tmp_dist) 
            self.renew_neighbor(X[index],index,y[index])
            tmp_dist[index] = float('inf')
            count += 1
    
        # ALL SAFE
        labels = np.array(self.nearest_neighbor_label)
        if len(labels[labels==minlabel]) == k: 
            safe = 1
      
            # Expand ALL SAFE AREA
            index = np.argmin(tmp_dist) 
            label = y[index]
            while label == minlabel:
                self.renew_neighbor(X[index],index,label)
                tmp_dist[index] = float('inf')
                index = np.argmin(tmp_dist) 
                label = y[index]

        # HALF SAFE
        elif len(labels[labels==minlabel]) > k/2: 
            safe = 2

            # If true. SHRINK HALF SAFE AREA
            if shrink_half == True:
                while self.nearest_neighbor_label[-1] != minlabel: 
                    self.del_neighbor()
                labels = np.array(self.nearest_neighbor_label)
                if len(labels[labels==minlabel]) == len(labels): 
                    safe = 1

            # If true. EXPAND HALF SAFE AREA        
            elif expand_half == True:
                index = np.argmin(tmp_dist) 
                label = y[index]
                while label == minlabel:
                    self.renew_neighbor(X[index],index,label)
                    tmp_dist[index] = float('inf')
                    index = np.argmin(tmp_dist) 
                    label = y[index]

        # Here we didn't consider about 3 max_all_safe 4 max_half_safe, we can add it if needed

        self.renew_paras(safe)
        self.neighbor_toarray()


def safe_areas(X,all_reps,y,minlabel=None,majlabel=None,k=3,shrink_half=False,expand_half=False):
    '''
    Generate all the representative points's area:
    if k neighbors of rep_point are all belonging to the minority class --> min safe area
    if more than k/2 neighbors of rep_point are belonging to the minority class --> min half safe area
    X: data
    all_reps: the list contain all the representative points generated by CURE
    y: label
    minlabel,majlabel: given the label of minority class and majority class, if None will be set from the dataset automatically (only work in binary classification case)
    k: should be an odd value, it will check k nearest neighbor of the representative points to decide safe or half safe, default 3
    shrink_half: if true it will try to shrink the half safe area to exclude the furthest majority class's point out of its neighbor until there is no change, default false 
    expand_half: if true it will try to expand the half safe area to contain more the nearest minority class's point into its neighbor until there is no chang, default false 
    '''
    areas = []
    min_all_safe_area = []
    min_half_safe_area = []
    for rep in all_reps:
        area = Area(rep)
        area.gen_safe_area(X,y,minlabel,majlabel,k,shrink_half,expand_half)
        areas.append(area)
        if area.safe == 1:
            min_all_safe_area.append(area)
        elif area.safe == 2:
            min_half_safe_area.append(area)
    return areas,min_all_safe_area,min_half_safe_area


def calc_num(min_all_safe_area,min_half_safe_area,minlabel):
    '''
    Calculate how many minority class' neighbors in total of all the all_safe_area and all the half_safe_area
    '''
    num_n_min_all_safe = 0
    num_n_min_half_safe = 0
    for area in min_all_safe_area:
        num_n_min_all_safe += area.num_neighbor
    for area in min_half_safe_area:
        neighbor = np.array(area.nearest_neighbor)
        label = np.array(area.nearest_neighbor_label)
        num_neighbor = len(neighbor[label==minlabel])
        num_n_min_half_safe += num_neighbor 
    return num_n_min_all_safe,num_n_min_half_safe

    
def calc_weight(total_num,min_all_safe_area,min_half_safe_area,all_safe_weight):
    '''
    Calculate the weighted number of new points for all safe area and half safe area
    Error will be reported when there is neither all safe area nor half safe area
    Commonly (no error case) w_all +  w_half = 1
    '''
    num_min_all_safe = len(min_all_safe_area)
    num_min_half_safe = len(min_half_safe_area)
    
    # Calculate the weight
    if num_min_half_safe == 0 and num_min_all_safe == 0:
        print('Error: There is not any safe area')
        return 0,0
    elif num_min_half_safe == 0: 
        w_all = 1
        w_half = 0
    elif num_min_all_safe == 0:
        w_all = 0
        w_half = 1
    else:
        w_all = all_safe_weight*num_min_all_safe/(all_safe_weight*num_min_all_safe+num_min_half_safe)
        w_half = num_min_half_safe/(all_safe_weight*num_min_all_safe+num_min_half_safe)
        
    total_num_all = math.ceil(total_num * w_all)
    total_num_half = total_num - total_num_all
    
    return total_num_all,total_num_half 
    

def generate(min_all_safe_area,min_half_safe_area,total_num,total_num_all,total_num_half,num_n_min_all,num_n_min_half,all_safe_weight,IR,all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,Gaussian_scale=None,minlabel=None,show=False):
    '''
    Return all generated synthetic points in all safe area and half safe area
    '''
    if minlabel == None:
        minlabel,_ = get_labels(y)
    
    if show == True:
        print(f"IR is {IR},need to generate {total_num} synthetic points, all safe weight is {all_safe_weight}")
        print(f"There are in total {len(min_all_safe_area)} all safe area, {len(min_half_safe_area)} half safe area")
        print(f"So generate ({all_safe_weight}*{len(min_all_safe_area)})/({all_safe_weight}*{len(min_all_safe_area)}+{len(min_half_safe_area)})={total_num_all} in all safe areas,{len(min_half_safe_area)}/({all_safe_weight}*{len(min_all_safe_area)}+{len(min_half_safe_area)})={total_num_half} in half safe areas")
    
    new_points = []
    for areas in [min_all_safe_area,min_half_safe_area]:
        if areas == min_all_safe_area:
            gen = all_safe_gen
            area_name = 'all safe area'
            total_num = total_num_all
            num_n = num_n_min_all

        elif areas == min_half_safe_area:
            gen = half_safe_gen
            area_name = 'half safe area'
            total_num = total_num_half
            num_n = num_n_min_half
        
        if len(areas) == 0:
            continue

        counter = 0

        for area in areas:
            neighbor = np.array(area.nearest_neighbor)
            label = np.array(area.nearest_neighbor_label)
            num_neighbor = len(neighbor[label==minlabel])   
            gen_num = int(total_num*(num_neighbor/num_n))
            

            para = G.check_parameter(half_safe_gen,minlabel,Gaussian_scale)
            new_points += list(gen(area,gen_num))
            counter += gen_num

            if show == True:
                print(f"{num_neighbor} minority neighbors in current area, so generate {gen_num} points around "+ area_name +F" of rep point {area.rep_point}")
        
        area_iter = itertools.cycle(areas)
        while counter < total_num:
            area = next(area_iter)
            new_points += list(gen(area,1))
            counter += 1 
            if show == True:
                print(f"generate 1 points around "+ area_name +F" of rep point {area.rep_point}")

    return np.array(new_points)


def oversampling(X,y,min_all_safe_area,min_half_safe_area,all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,Gaussian_scale=None,minlabel=None,majlabel=None,all_safe_weight=2,IR=1,show=False):
    '''
    Do the oversampling on safe areas 
    X: data
    y: label
    areas: the all Area instances list returned by cos.safe_areas() functions
    min_all_safe_area: the all safe Area instances list returned by cos.safe_areas() functions
    min_half_safe_area:  the all safe Area instances list returned by cos.safe_areas() functions
    all_safe_gen: the generator will be used in the all safe area, by defalut SMOTE, you can set it to any generator function from generate.py
    all_half_gen: the generator will be used in the half safe area, by defalut SMOTE, you can set it to any generator function from generate.py
    Gaussian_scale: the scale/standard deviation of Gaussian_Generator
    minlabel,majlabel: given the label of minority class and majority class, if None will be set from the dataset automatically (only work in binary classification case)
    all_safe_weight: the safe area's weight, the half safe area's weight is always 1, we just set the all safe area's weight is enough to control the ratio, by default 2
    IR: the expected imbalance ratio after the oversampling, by default 1
    show: show the generating process, by default False
    '''
    if all_safe_weight == None:
        all_safe_weight = 2
        
    if IR == None:
        IR = 1
    
    if minlabel == None and majlabel == None:
        minlabel,majlabel = get_labels(y)
    
    # To have a balance dataset, how many new points should we generate
    total_num = int(len(y[y==majlabel]) * IR)-len(y[y==minlabel]) 
    
    # How many neighbors(n) in all/half safe areas
    num_n_min_all,num_n_min_half = calc_num(min_all_safe_area,min_half_safe_area,minlabel)
    
    # How many points will be generate in all/half safe areas
    total_num_all,total_num_half = calc_weight(total_num,min_all_safe_area,min_half_safe_area,all_safe_weight)
    
    generated_points = generate(min_all_safe_area,min_half_safe_area,total_num,total_num_all,total_num_half,num_n_min_all,num_n_min_half,all_safe_weight,IR,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen,Gaussian_scale=Gaussian_scale,minlabel=minlabel,show=show)
   
    new_y = np.ones(len(generated_points))
    new_y = new_y * minlabel
    
    if len(generated_points) > 0 :
        X_generated = np.vstack((X,generated_points))
        y_generated = np.hstack((y,new_y))
    else:
        X_generated = X
        y_generated = y
        
    return X_generated,y_generated


def COS(X,y,N,c,alpha,linkage='cure_single',L=2,shrink_half=False,expand_half=False,all_safe_weight=2,all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,Gaussian_scale=None,IR=1,minlabel=None,majlabel=None,visualize=False):
    '''
    CURE(clustering and getting the representative points) -->
    safe area(Generate the safe areas around all the representative points) -->
    oversampling(Generate new points in safe areas)

    X: data
    y: label
    N: num_expected_clusters,how many clusters you want
    c: number of representative points in each cluster
    alpha: the given shrink parameter, the bigger alpha is the closer the representative points to the centroid of the cluster
     linkage: the linkage way in CURE
        if 'single' - calculate a nearest distance using representative points as the distance of two clusters, default value;
        if 'complete' - calculate a furthest distance using representative points as the distance of two clusters;  
        if 'average' - calculate a average distance using representative points as the distance of two clusters;
        if 'centroid' - calculate a distance using centorids as the distance of two clusters; 
        if 'ward' - calculate the variance if merging two clusters as the distance of two clusters;  
    L: the distance metric, L=1 the Manhattan distance, L=2 the Euclidean distance, by default L=2
    L: the distance metric will be used in CURE, L=1 the Manhattan distance, L=2 the Euclidean distance, by default L=2
    shrink_half: if true it will try to shrink the half safe area to exclude the furthest majority class's point out of its neighbor until there is no change, default false 
    expand_half: if true it will try to expand the half safe area to contain more the nearest minority class's point into its neighbor until there is no chang, default false 
    all_safe_weight: the safe area's weight, the half safe area's weight is always 1, we just set the all safe area's weight is enough to control the ratio, by default 2
    all_safe_gen: the generator will be used in the all safe area, by defalut SMOTE, you can set it to any generator function from generate.py
    all_half_gen: the generator will be used in the half safe area, by defalut SMOTE, you can set it to any generator function from generate.py
    Gaussian_scale: the scale/standard deviation of Gaussian_Generator
    IR: the expected imbalance ratio after the oversampling, by default 1
    minlabel,majlabel: given the label of minority class and majority class, if None will be set from the dataset automatically (only work in binary classification case)
    visualize: show the COS process, by default False
    '''
    if linkage in ['ward','single','complete','average','pyc_cure']:
        clusters,all_reps,num_reps = clusterings.clustering(X,y,N,c,alpha,linkage=linkage,L=L,minlabel=minlabel,majlabel=majlabel)
    else:
        # And define linkage == 'cure_ward'/'cure_single'....
        clusters,all_reps,num_reps = cure.Cure(X,N,c,alpha,linkage=linkage,L=L)
    areas,min_all_safe_area,min_half_safe_area = safe_areas(X,all_reps,y,minlabel=minlabel,majlabel=majlabel,shrink_half=shrink_half,expand_half=expand_half) 
    if visualize == True:
        print('Clusters:')
        V.show_clusters(clusters)
        print('Safe areas:')
        V.show_areas(X,y,min_all_safe_area,min_half_safe_area)

    X_generated,y_generated = oversampling(X,y,min_all_safe_area,min_half_safe_area,all_safe_gen=all_safe_gen,half_safe_gen=half_safe_gen,Gaussian_scale=Gaussian_scale,minlabel=minlabel,majlabel=majlabel,all_safe_weight=all_safe_weight,IR=IR,show=visualize)

    if visualize == True:
        print('Generated dataset:') 
        V.show_oversampling(X,y,X_generated,y_generated)
        plt.show()
        print('All:')
        V.show_cos(X,y,X_generated,y_generated,min_all_safe_area,min_half_safe_area,minlabel,majlabel)

    return X_generated,y_generated,len(min_all_safe_area),len(min_half_safe_area)