'''
@brief for reading the linkage from saved csv file deteremined by CPCC in experiment
'''


import pandas as pd
from utils import *


df = pd.read_csv('CPCC.csv')
df.columns = ['datasets'] + list(df.columns)[1:]
bests = df[['datasets','best']].values
linkages = {base_file(i[0]): i[1] for i in bests}
