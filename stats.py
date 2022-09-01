import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession
import os
#import sys
from scipy.sparse import csr_matrix
import geopandas as gpd
import pickle
from pathlib import Path
import argparse
from pyspark.sql.functions import expr, to_timestamp, col, unix_timestamp, first, sum, count
from pyspark.sql.types import *
from pyspark.sql.types import LongType, IntegerType
# importing module
import logging
 
#

# Datapath file CSV (dataset)
pathCSV = 'data/trips2323_new.csv'

# Path ShapeFile
pathShapeF = 'data/shapefile/shapefile_2323.shp'

# Path for results
pathRes = 'results/'

# Skip division per 0
np.seterr(divide='ignore', invalid='ignore')



def preprocessing(df):
    #df_cache = df.spark.cache()
    df = df.rename(columns={'from_zone':'from_zone_fid', 'from_timedate':'from_timedate_gmt', 'to_zone':'to_zone_fid', 'to_timedate':'to_timedate_gmt'})

    df = df.dropna(subset=['to_zone_fid'])
    fill = {'from_zone_fid': -1}
    df = df.fillna(value=fill)


    # Initial Filter
    # (df.triptime_s < 3278)
    df = df.loc[ (df.tripdistance_m > 0) & (df.stoptime_s > 0) & (df.stoptime_s < 604800)]

    #df = df.sort_values(by=['from_timedate_gmt', 'to_timedate_gmt'])
    #df = df.reset_index(drop = True)

    df_spark = df.to_spark()

    df_spark = df_spark.withColumn("from_timedate_gmt", expr("substring(from_timedate_gmt, 1, length(from_timedate_gmt)-3)"))
    df_spark = df_spark.withColumn("to_timedate_gmt", expr("substring(to_timedate_gmt, 1, length(to_timedate_gmt)-3)"))
    #df_spark = df_spark.withColumn("from_timedate_gmt", df_spark.from_timedate_gmt.substr(1, 6))

    df = df_spark.to_koalas()

    #df_cache.spark.unpersist()

    return df

def square(s):
    return s/60




if __name__ == '__main__':
    
    spark = SparkSession.builder\
        .master("local[8]")\
        .appName("Pyspark")\
        .config('spark.ui.port', '4050')\
        .config('spark.executor.cores', "100")\
        .getOrCreate()
    spark.catalog.clearCache()
    
    
    # Upload CSV datas
    df = ks.read_csv(pathCSV, sep=';', dtype = {'from_timedate':str, 'to_timedate':str})
    #df = df.spark.cache()
    df = df.spark.repartition(30)
    df = preprocessing(df)

#    print(df.head())
#    exit()

    print('-----------OUTPUT------------')

    #df_cache = df.spark.cache()
    df = df.rename(columns={'from_zone':'from_zone_fid', 'from_timedate':'from_timedate_gmt', 'to_zone':'to_zone_fid', 'to_timedate':'to_timedate_gmt'})

    df = df.dropna(subset=['to_zone_fid'])
    fill = {'from_zone_fid': -1}
    df = df.fillna(value=fill)
# Initial Filter
# (df.triptime_s < 3278)
    df = df.loc[ (df.tripdistance_m > 0) & (df.stoptime_s > 0) & (df.stoptime_s != 31422298)]
    df.stoptime_s = df.stoptime_s.apply(square)
    #test = df.stoptime_s.to_numpy()
    
    #ax = 
    df.stoptime_s.plot.kde(bw_method=3)  
    #fig = ax.get_figure()
    #fig.savefig('figure.png')
    #print(df.stoptime_s)#.plot.kde(bw_method=3) 

# Eliminare veicolo con filtro massimo e verificare se cambia qualcosa (anomalia o verità)
# Altrimenti porre 1 settimana come filtro massimo
# sistematicità nei parcheggi? ripetizione? dipendenza temporale  
# Heatmap?