'''
@brief area expand from represnetative points
'''
import numpy as np

from utils import get_labels
from .nearest_neighbor import nn_kd

class Area():
    '''
    @brief The area expand from representative points
    @param
        .rep_point: the representative point which the area generated from 
        .nearest_neighbor: neighbors in area
        .nearest_neighbor_index: neighbors' index
        .nearest_neighbor_label: neighbors' label
        .num_neighbor: number of neighbors in this area
        .radius: the radius from the representative point to its furthest neighbor)
        .safe: safety arrtibute of this area, 1:min_all_safe 2:min_half_safe 3:maj_all_safe 4:maj_half_safe
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
        self.num_min = 0 
        
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
        @brief Remove the last neighbor
        @note Will be used when shrink the half safe area
        '''
        self.nearest_neighbor = self.nearest_neighbor[:-1]
        self.nearest_neighbor_index = self.nearest_neighbor_index[:-1]
        self.nearest_neighbor_label = self.nearest_neighbor_label[:-1]
        self.nearest_neighbor_dist = self.nearest_neighbor_dist[:-1]

    def renew_paras(self,safe,minlabel):
        '''
        @note Will be called after generate the safe area
        '''
        self.num_neighbor = len(self.nearest_neighbor)
        self.safe = safe 
        self.radius = self.nearest_neighbor_dist[-1]
        self.num_min = len(self.nearest_neighbor_label[self.nearest_neighbor_label == minlabel])
        
        
    def gen_safe_area(self,X,y,tree,minlabel=None,majlabel=None,k=3):
        '''
        @brief Generate safe/half safe area
        @detail If k neighbors of rep_point are all belonging to the minority class --> min safe area
                If more than k/2 neighbors of rep_point are belonging to the minority class --> min half safe area
        k: should be an odd value, it will check k nearest neighbor of the representative points to decide safe or half safe, 3 by default 
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
    
        labels = self.nearest_neighbor_label
        # ALL SAFE
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
            # EXPAND HALF SAFE AREA
            index,label,dist = self.expand(tree,y,k,add)
            while any(label == minlabel):
                self.append_neighbor(X[index],index,label,dist)
                k += add
                index,label,dist = self.expand(tree,y,k,add) 
                   
            # SHRINK HALF SAFE AREA
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
            while cnt < 3:
                index,label,dist = self.expand(tree,y,k,add)
                self.append_neighbor(X[index],index,label,dist)
                k += add
                labels = self.nearest_neighbor_label
                if len(labels[labels==minlabel]) > k/2: 
                    # Become a half safe area
                    safe = 2
                    break
                cnt += 1
                
            # EXPAND HALF SAFE AREA
            if safe != 0 :
                index,label,dist = self.expand(tree,y,k,add)
                while any(label == minlabel):
                    self.append_neighbor(X[index],index,label,dist)
                    k += add
                    index,label,dist = self.expand(tree,y,k,add)
                    
                while self.nearest_neighbor_label[-1] != minlabel: 
                    self.del_neighbor()
                    
        self.renew_paras(safe,minlabel)

    def expand(self,tree,y,k,add):
        inds,dists = nn_kd([self.rep_point],k+add,tree)
        index = inds[k:k+add]
        dist = dists[k:k+add]
        label = y[index]
        return index,label,dist
    

def pop_areas(X,tree,all_reps,y,k=3,minlabel=None,majlabel=None):
    '''
    @brief Populate area surround representative points's area:
    '''
    areas = []
    min_all_safe_area = []
    min_half_safe_area = []
    for rep in all_reps:
        area = Area(rep)
        area.gen_safe_area(X,y,tree,minlabel,majlabel,k)
        areas.append(area)
        if area.safe == 1:
            min_all_safe_area.append(area)
        elif area.safe == 2:
            min_half_safe_area.append(area)
    return areas,min_all_safe_area,min_half_safe_area