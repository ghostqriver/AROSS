'''
@brief AROS's kd-tree nearest neighbor implementation
'''
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

from cluster import clustering
from utils import visualize as V
from utils import get_labels
from utils.analyze import min_proportion
from .nearest_neighbor import create_kd
from . import generate as G
from .area import *


def AROS(X,y,N,linkage='ward',alpha=0,L=2,IR=1,all_safe_weight=1,visualize=False):
    '''
    @brief AROS algorithm implementation
    @param 
        X: Data
        y: label
        N: n_clusters
        linkage: linkage of agglometative clustering
        alpha: shrink_rate of represents
        L: distance metric
        IR: expected_IR
    @return
    ''' 
    # Cluster the dataset and extract representatives
    minlabel,majlabel = get_labels(y)
    clusters,all_reps,_,labels = clustering(X,y,N,alpha,linkage,L)
    
    # Populate areas
    tree = create_kd(X)
    areas,min_all_safe_area,min_half_safe_area = pop_areas(X,tree,all_reps,y,minlabel=minlabel,majlabel=majlabel) 
    
    # For analyzing min neighbors in areas
    # safe_min_neighbors,all_min_neighbors = min_proportion(y,minlabel,min_all_safe_area,min_half_safe_area)
    
    if visualize:
        print('Clusters:')
        V.show_clusters_(X,labels,y,all_reps)
        print('Safe areas:')
        V.show_areas(X,y,min_all_safe_area,min_half_safe_area)
    
    # Oversample areas
    X_generated,y_generated = oversampling(X,tree,y,min_all_safe_area,min_half_safe_area,minlabel=minlabel,majlabel=majlabel,all_safe_weight=all_safe_weight,IR=IR,visualize=visualize)
    
    if visualize:
        print('Generated dataset:') 
        V.show_oversampling(X,y,X_generated,y_generated)
        plt.show()
        print('All:')
        V.show_cos(X,y,X_generated,min_all_safe_area,min_half_safe_area,minlabel,majlabel)
        
    return X_generated,y_generated#,safe_min_neighbors,all_min_neighbors


def oversampling(X,tree,y,min_all_safe_area,min_half_safe_area,minlabel=None,majlabel=None,all_safe_weight=1,IR=1,visualize=False):
    '''
    @brief Oversampling in areas    
    '''    
    if minlabel == None or majlabel == None:
        minlabel,majlabel = get_labels(y)
    
    # Synthetic instances should be generated to reach expected_IR
    total_num = int(len(y[y==majlabel]) * IR)-len(y[y==minlabel]) 
    
    # Number of min neighbors in safe/half-safe areas
    num_n_min_all,num_n_min_half = calc_num(min_all_safe_area,min_half_safe_area,minlabel)
    
    # Synthetic instances should be generate in safe/half-safe areas
    total_num_all,total_num_half = calc_weight(total_num,min_all_safe_area,min_half_safe_area,all_safe_weight)
    
    # Generate
    generated_points = generate(min_all_safe_area,min_half_safe_area,total_num,total_num_all,total_num_half,num_n_min_all,num_n_min_half,all_safe_weight,IR,tree=tree,y=y,minlabel=minlabel,visualize=visualize)
    # Append to dataset
    new_y = np.ones(len(generated_points))
    new_y = new_y * minlabel
    
    if len(generated_points) > 0 :
        X_generated = np.vstack((X,generated_points))
        y_generated = np.hstack((y,new_y))
    else:
        X_generated = X
        y_generated = y
        
    return X_generated,y_generated
    

def generate(min_all_safe_area,min_half_safe_area,total_num,total_num_all,total_num_half,num_n_min_all,num_n_min_half,all_safe_weight,IR,tree,y,minlabel=None,visualize=False):

    '''
    @return synthetic instances without label
    '''
    if minlabel == None:
        minlabel,_ = get_labels(y)
    
    if visualize:
        print(f"IR is {IR},need to generate {total_num} synthetic points, all safe weight is {all_safe_weight}")
        print(f"There are in total {len(min_all_safe_area)} all safe area, {len(min_half_safe_area)} half safe area")
        print(f"So generate ({all_safe_weight}*{len(min_all_safe_area)})/({all_safe_weight}*{len(min_all_safe_area)}+{len(min_half_safe_area)})={total_num_all} in all safe areas,{len(min_half_safe_area)}/({all_safe_weight}*{len(min_all_safe_area)}+{len(min_half_safe_area)})={total_num_half} in half safe areas")
    
    new_points = []
    gen = G.Gaussian_Generator
    for areas in [min_all_safe_area,min_half_safe_area]:
        if areas == min_all_safe_area:
            area_name = 'all safe area'
            total_num = total_num_all 
            num_n = num_n_min_all # number of min instances in safe area
            tree_ = None
        elif areas == min_half_safe_area:
            area_name = 'half safe area'
            total_num = total_num_half 
            num_n = num_n_min_half # number of min instances in half-safe area
            tree_ = tree
            
        if len(areas) == 0:
            continue

        counter = 0

        for area in areas:
            
            # Decide number to generate
            neighbor = np.array(area.nearest_neighbor)
            label = np.array(area.nearest_neighbor_label)
            num_neighbor = len(neighbor[label==minlabel])   
            gen_num = int(total_num*(num_neighbor/num_n))

            if visualize:
                print(f"{num_neighbor} minority neighbors in current area, so generate {gen_num} points around "+ area_name +F" of rep point {area.rep_point}")
            
            gen_points = list(gen(area,gen_num,tree=tree_,y_train=y,min_label=minlabel))
            new_points += gen_points
            gen_num_ = len(gen_points)
            counter += gen_num_
            
            if visualize and gen_num_!=gen_num:
                print(f"* Failed with generating {gen_num} points, only {gen_num_} points generated around "+ area_name +F" of rep point {area.rep_point}")
        
        # Distribute remaining synthetic instances to safe/half-safe areas
        if len(min_all_safe_area)>0:
            area_iter = itertools.cycle(min_all_safe_area)
            tree = None
        else:
            area_iter = itertools.cycle(areas)
        while counter < total_num:    
            area = next(area_iter)
            gen_points = list(gen(area,1,tree=tree_,y_train=y,min_label=minlabel))
            new_points += gen_points 
            counter += len(gen_points) 
            if visualize == True and len(gen_points)==1:
                print(f"generate 1 points around "+ area_name +F" of rep point {area.rep_point}")
    return np.array(new_points)


def calc_num(min_all_safe_area,min_half_safe_area,minlabel):
    '''
    @return number of min instances (duplicated allowed) in safe area and half-safe area
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
    @return number of synthetic instances should be generated in safe area and half-safe area
    '''
    num_min_all_safe = len(min_all_safe_area)
    num_min_half_safe = len(min_half_safe_area)
    
    # Calculate the weight
    if num_min_half_safe == 0 and num_min_all_safe == 0:
        print('Error: There is not any safe area')
        return 0,0
    elif num_min_half_safe == 0: 
        w_all = 1
    elif num_min_all_safe == 0:
        w_all = 0
    else:
        w_all = all_safe_weight*num_min_all_safe/(all_safe_weight*num_min_all_safe+num_min_half_safe)
        
    total_num_all = math.ceil(total_num * w_all)
    total_num_half = total_num - total_num_all
    
    return total_num_all,total_num_half 