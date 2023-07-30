'''
@brief decide linkage criteria by cophenetic distance
@author yizhi
'''

from scipy.cluster.hierarchy import single,ward,complete,average,cophenet
from scipy.spatial.distance import pdist


def cpcc(X):
    
    linkages = ['single','complete','average','ward']
    cond_m = pdist(X)
    
    # linkages
    single_ = single(cond_m)
    complete_ = complete(cond_m)
    average_ = average(cond_m)
    ward_ = ward(cond_m)
    
    # cophenetic correlation coefficient
    single_cp = cophenet(single_,cond_m)[0]
    complete_cp = cophenet(complete_,cond_m)[0]
    average_cp = cophenet(average_,cond_m)[0]
    ward_cp = cophenet(ward_,cond_m)[0]
    
    coeffs = [single_cp,complete_cp,average_cp,ward_cp]
    index_best = coeffs.index(max(coeffs))

    return linkages[index_best]