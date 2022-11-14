'''
For visualizing the dataset during the steps of COS
note:   because we always run the COS in the notebook(automatically show the figure without plt.show()), and there are some functions called by other visualization functions as a subplot
        in some functions there is not plt.show() after draw the figure, please modify the function or call the plt.show() in your own code line. 
'''

import matplotlib.pyplot as plt
import numpy as np

from . import cos
from . import generate as G


def show_2d_scatter(X,y,minlabel=None,majlabel=None):
    '''
    Show the scatter for a 2d(only two features, and features should be floats) dataset, unlimited for labels
    X: the value of data, in the n*2 np.array commonly
    y: labels of the dataset, in the 1d np.array commonly 
    '''
    if minlabel == None and majlabel ==None:
        minlabel,majlabel = cos.get_labels(y)
    plt.scatter(X[y==minlabel,0],X[y==minlabel,1],marker='*',c='blue',label='minority class')
    plt.scatter(X[y==majlabel,0],X[y==majlabel,1],marker='.',c='k',label='majority class')
    plt.legend()
    # plt.show()


def show_clusters(clusters):
    '''
    Show the scatter for a 2d dataset's cluster result
    clusters: the cluster object list from the CURE
    '''
    # plt.figure(figsize=(10,10))
    color_list = list(color_dict.keys())
    mod_ = len(color_list)
    for ind,cluster in enumerate(clusters):
        points = np.array(cluster.points)
        rep_points = np.array(cluster.rep_points)
        c_ind = (ind*3) % mod_
        plt.scatter(points[:,0],points[:,1],marker='.',c=color_list[c_ind])
        plt.scatter(rep_points[:,0],rep_points[:,1],marker='x',c=color_list[c_ind])
    plt.title('clusters')
    # plt.show()


def show_rep_points(X,y,clusters):
    '''
    Show the scatter for a 2d(only two features, and features should be floats) dataset, and all representative points get from the CURE algorithm
    X: the value of data, in the n*2 np.array commonly
    y: labels of the dataset, in the 1d np.array commonly 
    clusters: the cluster object list from the CURE
    '''
    labels = set(list(y))
    for label in labels:
        plt.scatter(X[y==label,0],X[y==label,1])
    for cluster in clusters:
        rep_points = np.array(cluster.rep_points)
        plt.scatter(rep_points[:,0],rep_points[:,1],marker='x',c='k')
    plt.title('representative points')
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



color_dict = {
            'black':                '#000000',
            'blue':                 '#0000FF',
            'blueviolet':           '#8A2BE2',
            'brown':                '#A52A2A',
            'burlywood':            '#DEB887',
            'cadetblue':            '#5F9EA0',
            'chartreuse':           '#7FFF00',
            'chocolate':            '#D2691E',
            'coral':                '#FF7F50',
            'cornflowerblue':       '#6495ED',
            'crimson':              '#DC143C',
            'cyan':                 '#00FFFF',
            'darkblue':             '#00008B',
            'darkcyan':             '#008B8B',
            'darkgoldenrod':        '#B8860B',
            'darkgray':             '#A9A9A9',
            'darkgreen':            '#006400',
            'darkkhaki':            '#BDB76B',
            'darkmagenta':          '#8B008B',
            'darkolivegreen':       '#556B2F',
            'darkorange':           '#FF8C00',
            'darkorchid':           '#9932CC',
            'darkred':              '#8B0000',
            'darksalmon':           '#E9967A',
            'darkseagreen':         '#8FBC8F',
            'darkslateblue':        '#483D8B',
            'darkslategray':        '#2F4F4F',
            'darkturquoise':        '#00CED1',
            'darkviolet':           '#9400D3',
            'deeppink':             '#FF1493',
            'deepskyblue':          '#00BFFF',
            'dimgray':              '#696969',
            'dodgerblue':           '#1E90FF',
            'firebrick':            '#B22222',
            'forestgreen':          '#228B22',
            'fuchsia':              '#FF00FF',
            'gainsboro':            '#DCDCDC',
            'gold':                 '#FFD700',
            'goldenrod':            '#DAA520',
            'gray':                 '#808080',
            'green':                '#008000',
            'greenyellow':          '#ADFF2F',
            'hotpink':              '#FF69B4',
            'indianred':            '#CD5C5C',
            'indigo':               '#4B0082',
            'khaki':                '#F0E68C',
            'lawngreen':            '#7CFC00',
            'lemonchiffon':         '#FFFACD',
            'lime':                 '#00FF00',
            'limegreen':            '#32CD32',
            'magenta':              '#FF00FF',
            'maroon':               '#800000',
            'mediumaquamarine':     '#66CDAA',
            'mediumblue':           '#0000CD',
            'mediumorchid':         '#BA55D3',
            'mediumpurple':         '#9370DB',
            'mediumseagreen':       '#3CB371',
            'mediumslateblue':      '#7B68EE',
            'mediumspringgreen':    '#00FA9A',
            'mediumturquoise':      '#48D1CC',
            'mediumvioletred':      '#C71585',
            'midnightblue':         '#191970',
            'mistyrose':            '#FFE4E1',
            'moccasin':             '#FFE4B5',
            'navy':                 '#000080',
            'olive':                '#808000',
            'olivedrab':            '#6B8E23',
            'orange':               '#FFA500',
            'orangered':            '#FF4500',
            'orchid':               '#DA70D6',
            'palegoldenrod':        '#EEE8AA',
            'palegreen':            '#98FB98',
            'paleturquoise':        '#AFEEEE',
            'palevioletred':        '#DB7093',
            'papayawhip':           '#FFEFD5',
            'peachpuff':            '#FFDAB9',
            'peru':                 '#CD853F',
            'pink':                 '#FFC0CB',
            'plum':                 '#DDA0DD',
            'powderblue':           '#B0E0E6',
            'purple':               '#800080',
            'red':                  '#FF0000',
            'rosybrown':            '#BC8F8F',
            'royalblue':            '#4169E1',
            'saddlebrown':          '#8B4513',
            'salmon':               '#FA8072',
            'sandybrown':           '#FAA460',
            'seagreen':             '#2E8B57',
            'seashell':             '#FFF5EE',
            'sienna':               '#A0522D',
            'skyblue':              '#87CEEB',
            'slateblue':            '#6A5ACD',
            'slategray':            '#708090',
            'springgreen':          '#00FF7F',
            'steelblue':            '#4682B4',
            'tan':                  '#D2B48C',
            'teal':                 '#008080',
            'thistle':              '#D8BFD8',
            'tomato':               '#FF6347',
            'turquoise':            '#40E0D0',
            'violet':               '#EE82EE',
            'yellow':               '#FFFF00'
            }
