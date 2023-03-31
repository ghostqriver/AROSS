'''
@brief COS's kd-tree nearest neighbor implementation

@example
'''
from . import generate as G
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from COS_Funcs.cluster import clustering
from COS_Funcs.utils import visualize as V
from COS_Funcs.utils import get_labels
from COS_Funcs.cos.nearest_neighbor import nn_kd,create_kd

def COS(X,y,N,c,alpha,linkage='cure_single',L=2,shrink_half=False,expand_half=True,all_safe_weight=2,all_safe_gen=G.Smote_Generator,half_safe_gen=G.Smote_Generator,Gaussian_scale=None,IR=1,visualize=False):
    
    minlabel,majlabel = get_labels(y)
    clusters,all_reps,num_reps = clustering(X,y,N,c,alpha,linkage,L)
    areas,min_all_safe_area,min_half_safe_area = safe_areas(X,all_reps,y,shrink_half=shrink_half,expand_half=expand_half,minlabel=minlabel,majlabel=majlabel) 
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
    
def safe_areas(X,all_reps,y,shrink_half=False,expand_half=True,k=3,minlabel=None,majlabel=None):
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
    tree = create_kd(X)
    for rep in all_reps:
        area = Area(rep)
        area.gen_safe_area(X,y,tree,minlabel,majlabel,shrink_half,expand_half,k)
        areas.append(area)
        if area.safe == 1:
            min_all_safe_area.append(area)
        elif area.safe == 2:
            min_half_safe_area.append(area)
    return areas,min_all_safe_area,min_half_safe_area

class Area():
    '''
    @brief The area around the representative point
    @paras
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
        self.nearest_neighbor_dist = []
        
        self.num_neighbor = 0
        self.radius = 0
        self.safe = 0 # 0 no_safe 1 min_all_safe 2 min_half_safe 

    def append_neighbor(self,data,index,label,dist):
        '''
        @brief Append the neighbor info to arrays
        @note Will be called when expand the all safe area
        '''
        self.nearest_neighbor = np.append(self.nearest_neighbor,data,axis=0)
        self.nearest_neighbor_index = np.append(self.nearest_neighbor_index,index)
        self.nearest_neighbor_label = np.append(self.nearest_neighbor_label,label)
        self.nearest_neighbor_dist = np.append(self.nearest_neighbor_dist,dist)

    def del_neighbor(self):
        '''
        @brief Remove the final neighbor
        @note Will be used when shrink the half safe area
        '''
        self.nearest_neighbor = self.nearest_neighbor[:-1]
        self.nearest_neighbor_index = self.nearest_neighbor_index[:-1]
        self.nearest_neighbor_label = self.nearest_neighbor_label[:-1]
        self.nearest_neighbor_dist = self.nearest_neighbor_dist[:-1]

    def renew_paras(self,safe):
        '''
        @note Will be called after generate the safe area
        '''
        self.num_neighbor = len(self.nearest_neighbor)
        self.safe = safe 
        # the distance from rep_point to farthest neighbor in the area
        self.radius = self.nearest_neighbor_dist[-1]

    def gen_safe_area(self,X,y,tree,minlabel=None,majlabel=None,shrink_half=True,expand_half=True,k=3):
        '''
        @brief Generate safe/half safe area
        @detail If k neighbors of rep_point are all belonging to the minority class --> min safe area
                If more than k/2 neighbors of rep_point are belonging to the minority class --> min half safe area
        k: should be an odd value, it will check k nearest neighbor of the representative points to decide safe or half safe, default 3
        shrink_half: if true it will try to shrink the half safe area to exclude the furthest majority class's point out of its neighbor until there is no change, default false, after shrink if satisfied all safe condition, then it will set that half safe area to all safe area 
        expand_half: if true it will try to expand the half safe area to contain more the nearest minority class's point into its neighbor until there is no chang, default false 
        '''
        safe = 0

        if minlabel == None or majlabel ==None:
            minlabel,majlabel = get_labels(y)

        # Get k neighbors  
        inds,dists = nn_kd([self.rep_point],k,tree)
        self.nearest_neighbor = X[inds]
        self.nearest_neighbor_index = inds
        self.nearest_neighbor_label = y[inds]
        self.nearest_neighbor_dist = dists
    
        # ALL SAFE
        labels = self.nearest_neighbor_label
        if len(labels[labels==minlabel]) == k: 
            safe = 1
            add = 1
            # Expand ALL SAFE AREA
            index,label,dist = self.expand(tree,y,k,add)
            while label == minlabel:
                self.append_neighbor(X[index],index,label,dist)
                k += add
                index,label,dist = self.expand(tree,y,k,add)

        # HALF SAFE
        elif len(labels[labels==minlabel]) > k/2: 
            safe = 2
            add = 2
            # If true. EXPAND HALF SAFE AREA
            if expand_half == True:
                index,label,dist = self.expand(tree,y,k,add)
                while any(label == minlabel):
                    self.append_neighbor(X[index],index,label,dist)
                    k += add
                    index,label,dist = self.expand(tree,y,k,add)
      
            # If true. SHRINK HALF SAFE AREA
            if shrink_half == True:
                while self.nearest_neighbor_label[-1] != minlabel: 
                    self.del_neighbor()
                labels = self.nearest_neighbor_label
                # Check whether all safe
                if len(labels[labels==minlabel]) == len(labels): 
                    safe = 1
        
        # Not SAFE            
        else:
            add = 2
            cnt = 0
            while cnt < 5:
                index,label,dist = self.expand(tree,y,k,add)
                self.append_neighbor(X[index],index,label,dist)
                k += add
                labels = self.nearest_neighbor_label
                if len(labels[labels==minlabel]) > k/2: 
                    # Become a half safe area
                    safe = 2
                    break
                cnt += 1
            # If true. EXPAND HALF SAFE AREA
            if safe != 0 and expand_half == True:
                index,label,dist = self.expand(tree,y,k,add)
                while any(label == minlabel):
                    self.append_neighbor(X[index],index,label,dist)
                    k += add
                    index,label,dist = self.expand(tree,y,k,add)
                    
            if safe != 0 and shrink_half == True:
                while self.nearest_neighbor_label[-1] != minlabel: 
                    self.del_neighbor()
                # Impossible to be all safe

        self.renew_paras(safe)

    def expand(self,tree,y,k,add):
        inds,dists = nn_kd([self.rep_point],k+add,tree)
        index = inds[k:k+add]
        dist = dists[k:k+add]
        label = y[index]
        return index,label,dist
        
        
def calc_num(min_all_safe_area,min_half_safe_area,minlabel):
    '''
    Calculate how many minority class' neighbors in total of all the all_safe_area and all the half_safe_area
    '''
    num_n_min_all_safe = 0
    num_n_min_half_safe = 0
    for area in min_all_safe_area:
        num_n_min_all_safe += area.num_neighbor
    for area in min_half_safe_area:
        neighbor = area.nearest_neighbor
        label = area.nearest_neighbor_label
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

