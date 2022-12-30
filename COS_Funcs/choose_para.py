import numpy as np
import scipy
import math


def silhouette(model,X,max_cluster=None):
    s_scores = {} 
    
    max_s_score = -1
    best_cluster = 2
    
    if max_cluster == None:
        max_cluster = math.ceil(len(X)/2)
#         max_cluster = math.ceil(np.sqrt(len(X)/2))

    for n in range(2,max_cluster): 
        agg = model(n_clusters=n).fit(X)
        labels = agg.labels_
        s_score = silhouette_score(X, labels)
        s_scores[n] = (s_score) 
        if s_score>max_s_score:
            max_s_score = s_score
            best_cluster = n
    
    plt.figure(figsize=(12,8))
    plt.plot(list(s_scores.keys()),list(s_scores.values()))
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Value')
    plt.show()
    
    return best_cluster

def Calinski_Harabasz(model,X,max_cluster=None):
    s_scores = {} 
    
    max_s_score = -1
    best_cluster = 2
    
    if max_cluster == None:
        max_cluster = math.ceil(len(X)/2)
#         max_cluster = math.ceil(np.sqrt(len(X)/2))

    for n in range(2,max_cluster): 
        agg = model(n_clusters=n).fit(X)
        labels = agg.labels_
        s_score = calinski_harabasz_score(X, labels)
        s_scores[n] = (s_score) 
        if s_score>max_s_score:
            max_s_score = s_score
            best_cluster = n
    
    plt.figure(figsize=(12,8))
    plt.plot(list(s_scores.keys()),list(s_scores.values()))
    plt.title('Calinski harabasz score ')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski harabasz  Value')
    plt.show()
    
    return best_cluster

def Davies_Bouldin(model,X,max_cluster=None):
    s_scores = {} 
    
    min_s_score = np.inf
    best_cluster = 2
    
    if max_cluster == None:
        max_cluster = math.ceil(len(X)/2)
#         max_cluster = math.ceil(np.sqrt(len(X)/2))

    for n in range(2,max_cluster): 
        agg = model(n_clusters=n).fit(X)
        labels = agg.labels_
        s_score = davies_bouldin_score(X, labels)
        s_scores[n] = (s_score) 
        if s_score<min_s_score:
            min_s_score = s_score
            best_cluster = n
    
    plt.figure(figsize=(12,8))
    plt.plot(list(s_scores.keys()),list(s_scores.values()))
    plt.title('Davies Bouldin score ')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies bouldin Value')
    plt.show()
    
    return best_cluster

def choose_c(cluster,minlabel,majlabel):
    
    P = sum(cluster.labels == minlabel)# Number of instances in the minority class
    N = cluster.num # Number of instances in the cluster
    if P ==0:
        return 0
    
    # print('Minority class:',P,end=', ')
    # print('all instances:',N,end=', ')
    
    n1 = P**2/N

    majclass = cluster.points # Here majclass is all instances in the cluster
    sigma = np.var(majclass) # Strandard deviation of the majority class  # square root variance formula

    Zalpha = scipy.stats.norm.ppf(.05) # The critical value of the Z test at the significance level α
    epsilone =  pow(10,-4) # acceptable tolerance error that can be adjusted as required 10 power -4

    n2= (N*Zalpha*epsilone*sigma)/((N*epsilone**2)+ (Zalpha*epsilone*sigma**2))

    pr = n2/n1

    M = 1.5

    if pr < 1:
         size = n1
    elif pr > M:
         size = n1*M
    else:
         size = n2

#     if size > N:
#         size = N
    
    # print('extract representative points',math.ceil(size))
        
    return math.ceil(size)

# def choose_N():
    
    
    
#     return N



# def choose_alpha():
    
    
    
#     return alpha

