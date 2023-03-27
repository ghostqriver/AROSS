figsize = (7,6)

rep_color = 'k'
rep_mark = 'x'
rep_label = 'representative points'
rep_size = 35

min_mark = '.'
min_size = 50
min_color ='blue'
min_label = 'minority class'
   
maj_mark = '.'
maj_size = 40
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import colorir
from COS_Funcs.cos import cos
from COS_Funcs.cos import generate as G


def show_2d_scatter(X,y=None):
    '''
    @brief Show the scatter for a 2D dataset
    '''
    if y is None: 
        plt.scatter(X[:,0],X[:,1],marker=maj_mark,c=maj_color)
        plt.show()
        return 
    minlabel,majlabel = cos.get_labels(y)
    plt.figure(figsize=figsize)
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size)
    plt.legend()
    # plt.show()

def show_clusters(clusters):
    '''
    @brief Show the clustering result 2D dataset
    @para 
        clusters: the list contain cluster objects
    '''
    plt.figure(figsize=figsize)
    point_colors_ = itertools.cycle(point_colors)
    for cluster in clusters:
        point_color = next(point_colors_)
        rep_color = colorir.PolarGrad([point_color,'#000000'],)
        rep_color = rep_color.n_colors(4)[1]
        points = np.array(cluster.points)
        rep_points = np.array(cluster.rep_points)
        plt.scatter(points[:,0],points[:,1],marker=point_mark,c=point_color,s=point_size)
        plt.scatter(rep_points[:,0],rep_points[:,1],marker=rep_mark,c=rep_color,s=rep_size)
    plt.title('clusters')
    # plt.show()

def show_rep_points(X,y,clusters):
    '''
    @brief Show the representative points on the 2D dataset
    '''
    minlabel,majlabel = cos.get_labels(y)
    plt.figure(figsize=figsize)
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size)
    
    for cluster in clusters:
        if len(cluster.rep_points) > 0:
            rep_points = np.array(cluster.rep_points)
            plt.scatter(rep_points[:,0],rep_points[:,1],marker=rep_mark,c=rep_color,s=rep_size,alpha=alpha)
    plt.scatter(rep_points[0,0],rep_points[0,1],marker=rep_mark,c=rep_color,label=rep_label,s=rep_size)      
    plt.legend()

    # plt.show()


def show_clusters_rep_points(X,y,clusters):
    '''
    Show the figure of show_clusters and show_rep_points
    '''
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    show_clusters(clusters)
    plt.subplot(1,2,2)
    show_rep_points(X,y,clusters)


def show_areas(X,y,min_all_safe_area,min_half_safe_area,minlabel=None,majlabel=None):
    '''
    Show the minority class's safe areas (black circle: all safe, brown circle: half safe)
    X: the value of data, in the n*2 np.array commonly
    y: labels of the dataset, in the 1d np.array commonly 
    min_all_safe_area: the all safe Area instances list returned by cos.safe_areas() functions
    min_half_safe_area:  the all safe Area instances list returned by cos.safe_areas() functions
    minlabel,majlabel: given the label of minority class and majority class, if None will be set from the dataset automatically (only work in binary classification case)
    '''
    if minlabel == None and majlabel ==None:
            minlabel,majlabel = cos.get_labels(y)

    plt.figure(figsize=(10,10))
    # The original dataset
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker=min_mark,c=min_color,label=min_label,s=min_size)
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker=maj_mark,c=maj_color,label=maj_label,s=maj_size)

    # The areas
    plt.plot(X[0,0],X[0,1], c='k',label='all safe area')
    plt.plot(X[0,0],X[0,1], c='brown',label='half safe area')
    
    for area in min_all_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],marker=rep_mark,c=rep_color,label=rep_label,s=rep_size) 
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c='k')
        plt.plot(x, y2, c='k')

        
    for area in min_half_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],marker=rep_mark,c=rep_color,label=rep_label,s=rep_size) 
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c='brown')
        plt.plot(x, y2, c='brown')

    plt.legend()
    plt.show()


def draw_cycle(center,r,c='k'):

        x = np.linspace(center[0] - r, center[0] + r, 5000)
        y1 = np.sqrt(abs(r**2 - (x - center[0])**2)) + center[1]
        y2 = -np.sqrt(abs(r**2 - (x - center[0])**2)) + center[1]
        plt.plot(x, y1, c)
        plt.plot(x, y2, c)

def show_single_area(area,new_points=[],circle_c='k',minlabel=None,majlabel=None):
    
    if minlabel == None and majlabel ==None:
        minlabel,majlabel = G.get_label_in_areas(area)

    min_neighbor = area.nearest_neighbor[area.nearest_neighbor_label == minlabel]
    maj_neighbor = area.nearest_neighbor[area.nearest_neighbor_label == majlabel]
    plt.scatter(min_neighbor[:,0],min_neighbor[:,1],marker='*',c='blue',label = 'minority neighbors')
    plt.scatter(maj_neighbor[:,0],maj_neighbor[:,1],marker='.',c='k',label = 'majority neighbors')

    plt.scatter(area.rep_point[0],area.rep_point[1],c='black',marker='x',label = 'rep_points')
    if len(new_points) > 0:
        plt.scatter(new_points[:,0],new_points[:,1],marker = '$\heartsuit$',c = 'red',label = 'synthetic samples',alpha = 0.5)
    draw_cycle(area.rep_point,area.radius,c=circle_c)


def show_oversampling(X,y,X_oversampled,y_oversampled):
    '''
    Can show the original sample and synthetic sample clearly only when all synthetic sample is directly attached to the end of original data,
    Eg. SMOTE, CURE_SMOTE, COS
    Or it will be confused with sequence of data.  
    '''
    origin_index = len(X)
    show_2d_scatter(X,y)
    plt.scatter(X_oversampled[origin_index:,0],X_oversampled[origin_index:,1],marker = '$\heartsuit$',c = 'red',label = 'synthetic samples',alpha = 0.5)
    plt.legend()
    # plt.show()


def show_cos(X,y,X_oversampled,y_oversampled,min_all_safe_area,min_half_safe_area,minlabel=None,majlabel=None):
    
    if minlabel == None and majlabel ==None:
            minlabel,majlabel = cos.get_labels(y)

    plt.figure(figsize=(10,10))
    # The original dataset
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker='*',c='blue',label='minority class')
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker='.',c='k',label='majority class')

    # The areas
    plt.plot(X[0,0],X[0,1], c='k',label='all safe area')
    plt.plot(X[0,0],X[0,1], c='brown',label='half safe area')
    
    for area in min_all_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],c='black',marker='x')
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c='k')
        plt.plot(x, y2, c='k')

        
    for area in min_half_safe_area:
        radius = area.radius
        rep_point = area.rep_point
        plt.scatter(rep_point[0],rep_point[1],c='black',marker='x')
        
        #draw the circle
        x = np.linspace(rep_point[0] - radius, rep_point[0] + radius, 5000)
        y1 = np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        y2 = -np.sqrt(abs(radius**2 - (x - rep_point[0])**2)) + rep_point[1]
        plt.plot(x, y1, c='brown')
        plt.plot(x, y2, c='brown')

    origin_index = len(X)
    plt.scatter(X_oversampled[origin_index:,0],X_oversampled[origin_index:,1],marker = '$\heartsuit$',c = 'red',label = 'synthetic samples',alpha = 0.5)

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
