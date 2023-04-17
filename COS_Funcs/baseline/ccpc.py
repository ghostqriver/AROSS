import pandas as pd

df = pd.read_csv('CPCC.csv')
df.columns = ['datasets'] + list(df.columns)[1:]
bests = df[['datasets','best']].values
linkages = {i[0]: i[1] for i in bests}
