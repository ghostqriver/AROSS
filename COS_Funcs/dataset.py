import pandas as pd
import glob


def get_labels(y):
    '''
    Return the minority class's label and majority class's label when given all labels y
    Only works well on binary dataset as yet
    return minlabel,int(majlabel)
    '''
    valuecounts=pd.Series(y).value_counts().index
    majlabel=valuecounts[0]
    minlabel=valuecounts[1:]
    if len(minlabel)==1:
        minlabel=int(minlabel[0])
    return minlabel,int(majlabel)
    
    
def read_dataset_info(path):
    dict_ls = []
    for file_name in glob.glob(path+'*.csv'):
        df = pd.read_csv(file_name)
        y = df.values[:,-1]
        
        dict = {}
        print(file_name)
        dict['dataset'] = file_name.split('.')[0]#.split('\\')[1]
        minlabel,majlabel = get_labels(y)
        
        dict['minority_class'] = 'Class' + str(minlabel)
        dict['majority_class'] = 'Class' + str(majlabel)
        dict['num_of_attributes'] = len(df.columns) - 1
        dict['num_of_minority_samples'] = len(y[y==minlabel])
        dict['num_of_majority_samples'] = len(y[y==majlabel])
        dict['imbalance_ratio'] = dict['num_of_majority_samples']/dict['num_of_minority_samples']
        
        dict_ls.append(dict)
    #     print(dict)

    dataset_df = pd.DataFrame.from_dict(dict_ls)
    dataset_df.to_csv(path+'dataset_description.csv')
    print('Save the dataset info into,',path+'dataset_description.csv')
    
