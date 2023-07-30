'''
@brief for analyzing the influence of parameter alpha
@author yizhi
'''

def collect_min_neighbors(areas,minlabel):
    '''
    @brief collect number of minority instances (non-repetitive) in the area
    '''
    min_set = set()
    for area in areas:
        # min_neighbors = area.nearest_neighbor_index[area.nearest_neighbor_index]
        min_set.update(area.nearest_neighbor_index[area.nearest_neighbor_label==minlabel])        
    return len(min_set)


def min_proportion(y,minlabel,min_all_safe_area,min_half_safe_area):
    '''
    @brief collect the proportion of minority instances be captured in areas
    '''
    min_neighbors = sum(y==minlabel)
    safe_min_neighbors = collect_min_neighbors(min_all_safe_area,minlabel)
    all_min_neighbors = collect_min_neighbors(min_all_safe_area+min_half_safe_area,minlabel)
    safe_min_neighbors = safe_min_neighbors/min_neighbors
    all_min_neighbors = all_min_neighbors/min_neighbors
    return safe_min_neighbors,all_min_neighbors