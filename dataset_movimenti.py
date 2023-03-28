# -*- coding: utf-8 -*-
"""dataset_movimenti.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LlAlZlF0dz8H46KEXVvev0ZmDt859cCj

Codice creato da Alice per creare il file csv con i movimenti dei veicoli e le soste.
Da questo file csv generato possiamo creare l'indice della sosta media in contemporanea
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='once')
from tqdm import tqdm 
#import matplotlib.pyplot as plt
#from matplotlib import rcParams
#import seaborn as sns
#import statistics
#plt.style.use("ggplot")
#import matplotlib
#matplotlib.rcParams['figure.figsize'] = (25, 10)

path = 'data/trips2323_mese.csv'
path_output = 'data/parking_flow_movement_one_month.csv'


df=pd.read_csv(path, sep=';')
df=df.loc[df.stoptime_s>(60*5)].reset_index(drop=True)
df['km']=df['tripdistance_m']/1000
df['vel_km_h']=(df['tripdistance_m']/df['triptime_s'])*3.6
df['end_stoptime'] = pd.to_datetime(df['to_timedate']) + pd.to_timedelta(df['stoptime_s'], unit='s')
df['day_of_week']=pd.to_datetime(df.to_timedate).dt.strftime("%A")
df['minuti']=df['stoptime_s']/60
df['ora']=pd.to_datetime(df.to_timedate).dt.hour
df['data']=pd.to_datetime(df.to_timedate).dt.date
df['to_zone'] = df['to_zone'].fillna(value=-1)
df['in_out'] = df['to_zone'].where(df['to_zone'] == -1, 0)
df['in_out'] = df['in_out'].where(df['from_zone'] >=0, 1)
df.sort_values(by='end_stoptime', inplace = True)
print(df)

exit()

arrivi=df.loc[:,['idterm','to_timedate','to_zone']]
arrivi['type']='arrived'
arrivi.rename(columns = {'to_timedate':'time'}, inplace = True)
partenze=df.loc[:,['idterm','end_stoptime','to_zone']]
partenze['type']='left'
partenze.rename(columns = {'end_stoptime':'time'}, inplace = True)
movimenti=pd.concat((arrivi, partenze))
movimenti['time']=pd.to_datetime(movimenti.time)
movimenti['mese']=movimenti.time.dt.month
movimenti['giorno']=movimenti.time.dt.day
movimenti['ora']=movimenti.time.dt.hour
movimenti['day_of_week']=movimenti.time.dt.strftime("%A")
movimenti['to_zone'] = movimenti['to_zone'].fillna(value=-1)
movimenti=movimenti.dropna().sort_values(by=['time']).reset_index(drop=True)
movimenti['data']=pd.to_datetime(movimenti.time).dt.date




movimenti['n_auto_park']=0
diz={}
for i in movimenti.to_zone.unique():
    diz[i]=0

for i in tqdm(range(0, len(movimenti)-1)):
    if movimenti.loc[i, 'type']=='arrived':
        diz[movimenti.loc[i, 'to_zone']]+=1
        movimenti.loc[i, 'n_auto_park']=diz[movimenti.loc[i, 'to_zone']]
    else:
        diz[movimenti.loc[i, 'to_zone']]-=1
        movimenti.loc[i, 'n_auto_park']=diz[movimenti.loc[i, 'to_zone']]
# 43 min

# Il codice richiede un upper bound per la data

movimenti=movimenti.loc[movimenti.data<= max(df['data'])].reset_index(drop=True)



movimenti.to_csv(path_output, sep=';')