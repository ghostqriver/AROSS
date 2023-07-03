figsize = (7,7)

rep_color = '#592626'
rep_mark = 'x'
rep_label = 'representative points'
rep_size = 45

min_mark = 'o'
# min_size = 15
min_size = 45
min_color ='blue'
min_label = 'minority class'
   
maj_mark = 's'
# maj_size = 9
maj_size = 29

maj_color = 'gray'
maj_label = 'majority class'

point_mark = '.'
point_size = 40
point_colors = ['#1F77B4',
                '#FF7F0E',
                '#2CA02C',
                '#D62728',
                '#9467BD',
                '#8C564B',
                '#E377C2',
                '#7F7F7F',
                '#BCBD22',
                '#17BECF']

alpha = 0.8
all_safe_color = 'k'
all_safe_line_style = 'dashed'
all_safe_label='all safe area'
half_safe_color = 'brown'
half_safe_label='half safe area'
half_safe_line_style = 'dashed'

grid_color = 'gray'
grid_line = 'dashed'
grid_line_width = 0.5

plot_line_color = '#3D3D3D'
plot_line_width = 0.9
vline_color = '#FFAA15'
vline_width = 0.9
vline_style = 'dashed'

syn_mark = '^'
syn_c = '#B22222'
syn_label = 'synthetic samples'
syn_alpha = 0.7
# syn_size = 7
syn_size = 17
import colorir
grad = colorir.PolarGrad(["#2304C0","FE6546","80F365","1B205F"])
colors = iter(grad.n_colors(7)) #next(colors)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from COS_Funcs.utils import get_labels
from COS_Funcs.cos import generate as G

def def_figure():
    plt.figure(figsize=figsize,dpi=200)
    # plt.grid(visible=True,color=grid_color, linestyle=grid_line, linewidth=grid_line_width)
    plt.yticks([])
    plt.xticks([])
    
def show_2d_scatter(X,y=None):
    '''
    @brief Show the scatter for a 2D dataset
    '''
    plt.figure(figsize=figsize,dpi=800)
    if y is None: 
        plt.scatter(X[:,0],X[:,1],marker=maj_mark,c=maj_color)
        plt.show()
        return 
    minlabel,majlabel = get_labels(y)
    def_figure()
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size,alpha=alpha)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size,alpha=alpha)
    plt.legend(loc='upper right')
    plt.show()

def show_clusters(clusters):
    '''
    @brief Show the clustering result 2D dataset
    @para 
        clusters: the list contain cluster objects
    '''
    def_figure()
    point_colors_ = itertools.cycle(point_colors)
    for cluster in clusters:
        point_color = next(point_colors_)
        rep_color = colorir.PolarGrad([point_color,'#000000'],)
        rep_color = rep_color.n_colors(4)[1]
        points = np.array(cluster.points)
        rep_points = np.array(cluster.rep_points)
        plt.scatter(points[:,0],points[:,1],marker=point_mark,c=point_color,s=point_size)
        if len(rep_points) > 0:
            last_rep = rep_points
            plt.scatter(rep_points[:,0],rep_points[:,1],marker=rep_mark,c=rep_color,s=rep_size)
    plt.scatter(last_rep[:,0],last_rep[:,1],marker=rep_mark,c=rep_color,s=rep_size,label=rep_label)
    plt.title('clusters')
    plt.legend()
    plt.show()

def show_clusters_(X,labels,y,all_reps):
    def_figure()    
    minlabel,majlabel = get_labels(y)
    clus_len = max(labels)
    point_colors_ = itertools.cycle(point_colors)
    for clus_id in range(clus_len+1):
        point_color = next(point_colors_)
        clus = X[labels==clus_id]
        ys = y[labels==clus_id]
        plt.scatter(clus[ys==minlabel,0],clus[ys==minlabel,1],marker=min_mark,s=min_size,c=point_color)
        plt.scatter(clus[ys==majlabel,0],clus[ys==majlabel,1],alpha=alpha,marker=maj_mark,s=maj_size,c=point_color)
    
    plt.scatter(clus[ys==minlabel,0],clus[ys==minlabel,1],marker=min_mark,s=min_size,c=point_color,label=min_label)
    plt.scatter(clus[ys==majlabel,0],clus[ys==majlabel,1],alpha=alpha,marker=maj_mark,s=maj_size,c=point_color,label=maj_label)
    plt.scatter(all_reps[:,0],all_reps[:,1],marker=rep_mark,c=rep_color,s=rep_size,alpha=alpha,label=rep_label)

    plt.legend(loc='upper right')
    plt.show()

def show_rep_points(X,y,clusters):
    '''
    @brief Show the representative points on the 2D dataset
    @para 
        clusters: the list contain cluster objects
    '''
    minlabel,majlabel = get_labels(y)
    def_figure()
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size)
    
    for cluster in clusters:
        if len(cluster.rep_points) > 0:
            rep_points = np.array(cluster.rep_points)
            plt.scatter(rep_points[:,0],rep_points[:,1],marker=rep_mark,c=rep_color,s=rep_size,alpha=alpha)
    plt.scatter(rep_points[0,0],rep_points[0,1],marker=rep_mark,c=rep_color,label=rep_label,s=rep_size)   
    plt.legend()
    plt.show()



def show_areas(X,y,min_all_safe_area,min_half_safe_area,):
    '''
    @brief Show the areas (black circle: all safe, brown circle: half safe)
    @para 
        min_all_safe_area: the all safe Area list returned by cos.safe_areas() functions
        min_half_safe_area: the all safe Area list returned by cos.safe_areas() functions
    '''

    minlabel,majlabel = get_labels(y)

    def_figure()
    # The original dataset
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size)

    # The areas
    plt.plot(X[0,0],X[0,1], c=all_safe_color, label='all safe area',linestyle=all_safe_line_style)
    plt.plot(X[0,0],X[0,1], c=half_safe_color, label='half safe area',linestyle=half_safe_line_style)
    
    for area in min_all_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],marker=rep_mark,c=rep_color,s=rep_size) 
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c=all_safe_color,linestyle=all_safe_line_style)
        plt.plot(x, y2, c=all_safe_color,linestyle=all_safe_line_style)

        
    for area in min_half_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],marker=rep_mark,c=rep_color,s=rep_size) 
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c=half_safe_color,linestyle=half_safe_line_style)
        plt.plot(x, y2, c=half_safe_color,linestyle=half_safe_line_style)

    plt.scatter(rep_point[0],rep_point[1],marker=rep_mark,c=rep_color,s=rep_size,label=rep_label) 
    plt.legend()
    plt.show()


def draw_cycle(center,r,c='k'):

    x = np.linspace(center[0] - r, center[0] + r, 5000)
    y1 = np.sqrt(abs(r**2 - (x - center[0])**2)) + center[1]
    y2 = -np.sqrt(abs(r**2 - (x - center[0])**2)) + center[1]
    plt.plot(x, y1, c,linestyles=all_safe_line_style)
    plt.plot(x, y2, c,linestyles=all_safe_line_style)

def show_single_area(area,new_points=[],circle_c='k',minlabel=None,majlabel=None,close_axis=True):
    
    plt.figure(figsize=(6,6))
    if minlabel == None and majlabel ==None:
        minlabel,majlabel = G.get_label_in_areas(area)

    min_neighbor = area.nearest_neighbor[area.nearest_neighbor_label == minlabel]
    maj_neighbor = area.nearest_neighbor[area.nearest_neighbor_label == majlabel]
    plt.scatter(min_neighbor[:,0],min_neighbor[:,1],marker=min_mark,c=min_color,label=min_label)
    plt.scatter(maj_neighbor[:,0],maj_neighbor[:,1],marker=maj_mark,c=maj_color,label=maj_label)

    plt.scatter(area.rep_point[0],area.rep_point[1],c=rep_color,marker=rep_mark,label=rep_label[:-1]    )
    if len(new_points) > 0:
        plt.scatter(new_points[:,0],new_points[:,1],marker=syn_mark,c=syn_c,label=syn_label,alpha=syn_alpha)
    draw_cycle(area.rep_point,area.radius,c=circle_c)
    if close_axis:
        plt.xticks([])
        plt.yticks([])
    plt.grid(visible=True,color=grid_color, linestyle=grid_line, linewidth=grid_line_width)
    plt.legend(loc=3)
    plt.show()


def show_oversampling(X,y,X_oversampled,y_oversampled):
    '''
    Can show the original sample and synthetic sample clearly only when all synthetic sample is directly attached to the end of original data,
    Eg. SMOTE, CURE_SMOTE, COS
    Or it will be confused with sequence of data.  
    '''
    origin_index = len(X)
    minlabel,majlabel = get_labels(y)
    def_figure()
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size,alpha=alpha)
    plt.scatter(X_oversampled[origin_index:,0],X_oversampled[origin_index:,1],marker=syn_mark,label=syn_label,alpha=syn_alpha,s=syn_size,c=syn_c)
    # plt.legend()
    # plt.show()


def show_cos(X,y,X_oversampled,y_oversampled,min_all_safe_area,min_half_safe_area,minlabel=None,majlabel=None):
    
    if minlabel == None and majlabel ==None:
            minlabel,majlabel = get_labels(y)
    def_figure()
    # The original dataset
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size)

    # The areas
    plt.plot(X[0,0],X[0,1], c=all_safe_color,label=all_safe_label,linestyle=all_safe_line_style)
    plt.plot(X[0,0],X[0,1], c=half_safe_color,label=half_safe_label,linestyle=half_safe_line_style)
    
    for area in min_all_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],c=rep_color,marker=rep_mark)
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c=all_safe_color,linestyle=all_safe_line_style)
        plt.plot(x, y2, c=all_safe_color,linestyle=all_safe_line_style)

        
    for area in min_half_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],c=rep_color,marker=rep_mark)
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c=half_safe_color,linestyle=half_safe_line_style)
        plt.plot(x, y2, c=half_safe_color,linestyle=half_safe_line_style)

    origin_index = len(X)
    plt.scatter(X_oversampled[origin_index:,0],X_oversampled[origin_index:,1],marker=syn_mark,c=syn_c,label=syn_label,alpha=syn_alpha)
    plt.legend()
    plt.show()

def show_baselines_areas(file_name):
    file_name = 'baselines/c10_alpha0.5_N30_kappa_random_forest_k10.xlsx'
    sheets = pd.ExcelFile(file_name).sheet_names[:-1] # exclude the final avg one
    score_list = []
    num_all_safe_list = []
    num_half_safe_list = []

    for sheet in sheets:
        df = pd.read_excel(file_name,sheet_name=sheet,index_col=0)
        dataset_name_list = df['cos'].index
        score_list.append(df['cos'].values)
        num_all_safe_list.append(df['all safe area'].values)
        num_half_safe_list.append(df['half safe area'].values)

    score_list = np.array(score_list)
    num_all_safe_list = np.array(num_all_safe_list)
    num_half_safe_list = np.array(num_half_safe_list)
    folds = list(range(1,len(sheets)+1))

    for index,dataset in enumerate(dataset_name_list):
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(folds,num_all_safe_list[:,index],label=dataset+'_all safe areas',c='k',linestyle='dashed')
        ax1.plot(folds,num_half_safe_list[:,index],label=dataset+'_half safe areas',c='brown',linestyle='dashed')
        ax1.set_xlabel(str(len(folds))+' folds')
        ax1.set_ylabel('Number of safe areas')
        ax1.legend(loc = 1)
        ax2 = ax1.twinx() 
        ax2.plot(folds,score_list[:,index],label=dataset+'_score',c='blue')
        for fold,score in zip(folds,score_list[:,index]):
            ax2.text(fold,score,str(round(score,2)),c='blue')
        ax2.legend(loc = 2)
        ax2.set_ylabel('score')
        plt.show()
