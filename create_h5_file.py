import pandas as pd
import h5py
import numpy as np
import pickle
import os
from tqdm import tqdm
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"


gcn = True

# Generator of calendar with specific timestamp (start/end='dd-mm-yyyy', freq='20min', '8hour', ...)
def calendar(start, end, freq):
    calendar = pd.date_range(start=start, end=end, freq=freq)
    return calendar

with open('results_one_month/features_matrix_sosta_media_contemporanea_60min.pickle', 'rb') as f:
    x = pickle.load(f)

#x_0 = x[0].todense()
tot=np.zeros((2322, 1))
for i in tqdm(range(1, len(x))):
  tot = np.append(tot,  x[i].todense(), axis=1)

print(tot.shape)
df = pd.DataFrame(tot)

print('prima dimensione', df.to_numpy().shape)
if gcn:
  #df = pd.read_csv('results/feat_8h.csv', header=None)
  feat = df.to_numpy().T
  print(np.sum(feat, axis=0).shape)
  exit()
  feat_resh = feat.reshape((tot.shape[1], tot.shape[0])) #54, 43))
  print('Dimensione gcn features', feat_resh.shape)
else:
  feat_resh = df.to_numpy().reshape((tot.shape[1], 54, 43))
  print('Dimensione cnn features', feat_resh.shape)


cal = calendar('2013-05-01 00:00:00', '2013-05-31 23:59:00', '1H')
print(len(cal))
cal_np = cal.to_list()
lista = list()
# replico lo stesso procedimento fatto in 'main_creation_matrix'
for i in range(0, len(cal_np)-1):
  lista.append(cal_np[i].strftime("%Y-%m-%d %H:%M"))

cal_np2 = np.array(lista)
cal_np2 = cal_np2.astype(np.dtype('|S16'))

hf = h5py.File('results_one_month/feat_GNN_sosta_media_contemporanea1h.h5', 'w')
hf.create_dataset('data', data=feat_resh)
# Salvo le date che fanno riferimento a ogni singola matrice
hf.create_dataset('date', data=cal_np2)

hf.close()