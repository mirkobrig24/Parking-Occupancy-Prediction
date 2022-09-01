import pandas as pd
import databricks.koalas as ks
import h5py
import numpy as np
import pickle
import os
from tqdm import tqdm
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"




# Definizione del calendario con intervallo temporale scelto (start/end='dd-mm-yyy', freq=intervallo di creazione di ogni data (es. '15min'))
def calendar(start, end, freq):
  calendar = ks.date_range(start=start, end=end, freq=freq)
  return calendar

with open('results/features_matrix_8H.pickle', 'rb') as f:
    x = pickle.load(f)

#x_0 = x[0].todense()
tot=np.zeros((2322, 1))
for i in tqdm(range(1, len(x))):
  tot = np.append(tot,  x[i].todense(), axis=1)

print(tot.shape)
df = pd.DataFrame(tot)

print('prima dimensione', df.to_numpy().shape)

#df = pd.read_csv('results/feat_8h.csv', header=None)
feat = df.to_numpy().T
print('seconda', feat.shape)
feat_resh = feat.reshape((tot.shape[1], tot.shape[0])) #54, 43))
print('terza', feat_resh.shape)
exit()


cal = calendar('2013-01-01 00:00:00', '2013-12-31 23:59:00', '1H')
print(len(cal))
cal_np = cal.to_list()
lista = list()
# replico lo stesso procedimento fatto in 'main_creation_matrix'
for i in range(0, len(cal_np)-1):
  lista.append(cal_np[i].strftime("%Y-%m-%d %H:%M"))

cal_np2 = np.array(lista)
cal_np2 = cal_np2.astype(np.dtype('|S16'))

hf = h5py.File('results/feat_GCN_8h.h5', 'w')
hf.create_dataset('data', data=feat_resh)
# Salvo le date che fanno riferimento a ogni singola matrice
hf.create_dataset('date', data=cal_np2)

hf.close()