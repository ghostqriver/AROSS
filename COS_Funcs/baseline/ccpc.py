import pandas as pd
import os
df = pd.read_csv('CPCC.csv')
df.columns = ['datasets'] + list(df.columns)[1:]
bests = df[['datasets','best']].values
linkages = {os.path.basename(i[0]): i[1] for i in bests}
