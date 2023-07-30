from . import agglomerative

def clustering(X,y,N,alpha,linkage,L=2):
    
    return agglomerative.Agglomerativeclustering(X,y,N,alpha,linkage,L)