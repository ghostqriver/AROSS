from COS_Funcs.cluster import cure
from COS_Funcs.cluster import cure_pyc
from COS_Funcs.cluster import agglomerative

def clustering(X,y,N,c,alpha,linkage='cure_single',L=2):
    
    num_expected_clusters = N

    if 'cure' in linkage:

        if linkage == 'cure_single' and L==2:
            # Most fast CURE with ccore 
            clusters,all_reps,num_reps = cure_pyc.Cure(X,y,N,c,alpha,L)
        else:
            # Slow but precise
            clusters,all_reps,num_reps = cure.Cure(X,num_expected_clusters,c,alpha,linkage,L)

    elif linkage in ['ward','single','complete','average']:
        clusters,all_reps,num_reps = agglomerative.Agglomerativeclustering(X,y,N,c,alpha,linkage,L)

    return clusters,all_reps,num_reps