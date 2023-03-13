import numpy as np

import cure_kdtree as cure_kdtree

X = np.array([[1,5],[2,6],[9,8],[5,4],[3,3],[10,7]])
cure_ = cure_kdtree.Cure(2,3,0.5,linkage='cure_single',L=2,visualize = False)
clusters,all_reps,num_reps = cure_.fit(X)
# V.show_clusters(clusters)