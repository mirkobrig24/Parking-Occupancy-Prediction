import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(action='once')
import statistics
import matplotlib
import h5py
import geopandas as gpd
import pysal.lib as ps
import esda.moran as moran
from pysal.model import spreg
import statistics
from pysal.explore import esda  # Exploratory Spatial analytics
from pysal.lib import weights  # Spatial weights

# Utilizzare la media e KNN=20

# Dati
f = h5py.File("results_one_month/feat_CNN_mean_time_1h.h5", 'r')
data = np.array(f['data'])
vec_mediana_i1=np.median(data, axis=0).reshape(2322,)
vec_media_i1=np.mean(data, axis=0).reshape(2322,)

#f = h5py.File("C:/Users/alice.ondei_quantyca/Desktop/Tesi/Dati/feat_sosta_media_contemporanea_1h.h5", 'r')
#data = np.array(f['data'])
#vec_mediana_i2=np.median(data, axis=0).reshape(2322,)
#vec_media_i2=np.mean(data, axis=0).reshape(2322,)

d={}
d['m_i1']=vec_mediana_i1
d['me_i1']=vec_media_i1
#d['m_i2']=vec_mediana_i2
#d['me_i2']=vec_media_i2
df = pd.DataFrame.from_dict(d)
df.to_csv("ind.csv", sep=';')


in_shp = 'data/shapefile/shapefile_2323.shp'
shapefile = gpd.read_file(in_shp)
shapefile = shapefile.set_crs('epsg:4326')

usmap_json_no_id = shapefile.to_json(drop_id=True)

#path='/content/drive/MyDrive/TESI/Codice/Dati/ind.csv'

data=pd.read_csv('ind.csv', sep=';')

#data
#print(data)

data.rename(columns={"Unnamed: 0": "FID"}, errors="raise", inplace=True)

merged =pd.merge(shapefile, data, on='FID', how='left')
merged=merged.fillna(0)

#Pesi

#shp_path = '/content/drive/MyDrive/TESI/Codice/shapefile/shapefile_2323.shp'
gdf = gpd.read_file(in_shp)

# metodo 1: matrice di contiguità Queen
w_queen = weights.contiguity.Queen.from_dataframe(gdf)
# metodo 2: matrice di contiguità Rook
w_rook = weights.contiguity.Rook.from_dataframe(gdf)
# metodo 3: matrice di contiguità a distanza fissa
w2 = weights.distance.KNN.from_dataframe(shapefile, k=12)
w2.transform = "R"

w = weights.distance.KNN.from_dataframe(shapefile, k=15)
w.transform = "R"

w1 = weights.distance.KNN.from_dataframe(shapefile, k=20)
w1.transform = "R"

d={}

pesi=[w_queen, w_rook, w2, w, w1]
indici=['m_i1', 'me_i1']#, 'm_i2', 'me_i2' ]
m=[]
g=[]
desc=[]
n=0
for metodo in pesi:
  for i in indici:
    m.append(round(esda.moran.Moran(merged[i], metodo).I,3))
    g.append(round(esda.geary.Geary(merged[i], metodo).C,3))
    desc.append(i+'_'+str(n))
  n+=1
df = pd.DataFrame({'desc': desc, 'moran': m, 'geary': g})
print(df)