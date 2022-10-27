from . import cure
from . import cos
from . import visualize as V
import numpy as np
import matplotlib.pyplot as plt

def Cure_(X,c,alpha,n1=None,n2=None):
    '''
    Same to the Cure algorithm, only difference is that this parameter choosing method will record how many noisy clusters are there
    in the clustering result in this merging iteration. The goal here is to check the longest part of some noise clusters(only few 
    point)
    n1: the lower bound of N(num_expected_clusters)
    n2: the upper bound of N(num_expected_clusters)
    c: number of representative points in each cluster
    alpha: the given shrink parameter, the bigger alpha is the closer the representative to the centroid
    '''
    if n1 == None:
        n1 = 10
    if n2 == None:
        n2 = int(len(X)/c)
    count_noise=[]
    cluster_num=[]

    clusters = cure.Cluster.gen_cluster(X)
    
    dist = cure.dist_matrix(X)
    
    num_clusters = len(clusters)
        
    num_expected_clusters = n1
    while(num_clusters > num_expected_clusters):
        
        neighbor1,neighbor2,min_dist = dist.nearest_neighbor()
            
        clusters[neighbor1].merge(clusters[neighbor2],c,alpha)
        
        dist.renew_matrix(clusters,neighbor1,neighbor2)
               
        # Drop the unused clusters' informations
        del(clusters[neighbor2])
        
        # Decrease the total number of clusters
        num_clusters=num_clusters-1
    
        if num_clusters>=n1 and num_clusters<=n2:
            count_ = 0
            for cluster in clusters:
                if cluster.num < c:
                      count_+=1
            count_noise.append(count_)
            cluster_num.append(num_clusters)

    return cluster_num,count_noise


def show_noise_number(cluster_num,count_noise,ind):  
    plt.plot(cluster_num,count_noise) 
    plt.scatter(cluster_num[ind],count_noise[ind],color='red')
    plt.xlabel('num_expected_clusters')
    plt.ylabel('the number of clusters which only have less point(<c)')
    plt.show()
    
    
def Noise_number(X,c,alpha,n1=None,n2=None,noise_clus_tolerate=10,show=False):
    cluster_num,count_noise = Cure_(X,c,alpha,n1,n2)
    bincount = np.bincount(count_noise) #bincount[0]:the 0's occurence times in counts
    max_count = 0 
    max_index = -1
    for num,i in enumerate(bincount): 
        if i>max_count and num<noise_clus_tolerate and num>0:
            max_count=i
            max_index=num
    print(f'When c={c} alpha={alpha}, the most (also the most probability) time occur noise cluster number is',max_index)
      
    ind=np.min(np.where(np.array(count_noise)==max_index)) 
    
    num_expected_clusters=cluster_num[ind]
    print('Here propose to choose num_expected_clusters=',num_expected_clusters)
    if show==True:
        show_noise_number(cluster_num,count_noise,ind)
    return num_expected_clusters


def para_choosing(X,y,n1=None,n2=None,minlabel=None,majlabel=None,shrink_half=False,expand_half=False,noise_clus_tolerate=10,show=False):       
    '''
    By iterate possible parameters(c,alpha) and have its best N, calculate the score (5*min_all_safe+2.5*min_half_safe)/num_rep to get the parameters with the best score
    X: data
    y: label
    n1: the lower bound of N(num_expected_clusters)
    n2: the upper bound of N(num_expected_clusters)
    minlabel,majlabel: given the label of minority class and majority class, if None will be set from the dataset automatically (only work in binary classification case)
    shrink_half: if true it will try to shrink the half safe area to exclude the furthest majority class's point out of its neighbor until there is no change, default false 
    expand_half: if true it will try to expand the half safe area to contain more the nearest minority class's point into its neighbor until there is no chang, default false 
    show: show the choosing process, by default False
    '''
    if minlabel == None and majlabel ==None:
        minlabel,majlabel = cos.get_labels(y)
    
    
    c_candidate=[3,5,7,9,11] 
    alpha_candidate=[0.2,0.3,0.4,0.5,0.6,0.7]
    k=3 
    
    max_score=0
    max_score_c=c_candidate[0]
    max_score_alpha=alpha_candidate[0]
    max_score_N=n1
    
    score_list=[]
    best_count_min_all_safe=0
    min_count_list=[]

    for c_ in c_candidate: #c:i
        for alpha_ in alpha_candidate: #alpha:j
            best_N=Noise_number(X,c_,alpha_,n1,n2,noise_clus_tolerate,show=show)
            if show == True:
                print(f'When c={c_} alpha={alpha_} it proposed the best N is {best_N}')
        
            _,all_reps,num_rep=cure.Cure(X,best_N,c_,alpha_)
            min_all_safe_area,min_half_safe_area = cos.safe_areas(X,all_reps,y,minlabel=minlabel,majlabel=majlabel,shrink_half=shrink_half,expand_half=expand_half)
            count_min_all_safe = len(min_all_safe_area)
            score = (count_min_all_safe*5 + len(min_half_safe_area)*2.5)/num_rep

            score_list.append(score)
            min_count_list.append(count_min_all_safe)
            if score>=max_score and count_min_all_safe>best_count_min_all_safe:
                max_score=score
                max_score_c=c_
                max_score_alpha=alpha_
                max_score_N=best_N
                best_count_min_all_safe=count_min_all_safe
            if show==True:
                V.show_areas(X,y,min_all_safe_area,min_half_safe_area)
                plt.show()
            print(80*"-")
    if show == True:
        print(f'Finally we choose c={max_score_c} alpha={max_score_alpha} N={max_score_N} to get the best score {max_score} with min all safe area {best_count_min_all_safe}')
    return max_score_c,max_score_alpha,max_score_N,score_list,min_count_list