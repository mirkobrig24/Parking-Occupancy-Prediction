import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
import pyspark.pandas as ps
import databricks.koalas as ks
import pyspark.pandas as ps
import databricks.koalas as ks
from pyspark.sql import SparkSession
#import sys
import pandas as pd
from scipy.sparse import csr_matrix
import geopandas as gpd
import pickle
from pathlib import Path
import argparse
from pyspark.sql.functions import expr, to_timestamp, col, unix_timestamp, first, sum, count, concat_ws, row_number, date_format,from_unixtime,lag, col, lead, to_date, unix_timestamp, datediff, lit, split
from pyspark.sql.types import *
from pyspark.sql.types import LongType, IntegerType
from pyspark.sql.window import Window
import timeit
os.chdir(Path(__file__).resolve().parent)

# Datapath file CSV (dataset)
#pathCSV = 'data/parking_flow_movement.csv'
pathCSV = 'data/parking_flow_movement_one_month.csv'


# Path ShapeFile
pathShapeF = 'data/shapefile/shapefile_2323.shp'

# Path for results
#pathRes = 'results/'
pathRes = 'results_one_month/'

# Skip division per 0
np.seterr(divide='ignore', invalid='ignore')

# Parameters to set type of matrix and value of timestamp
parser = argparse.ArgumentParser(description="Parameters to decide the matrix's type and date interval")
parser.add_argument("--mat", type=str, help="Select which matrix ('adj_mat', 'feat_mat', 'all')", default='feat_mat')
parser.add_argument("--date_start", type=str, help="Start date of the interval in the format 'mm/dd/yyyy' (to select all the dataset type 'all')", default='all')
parser.add_argument("--date_end", type=str, help="End date of the interval in the format 'mm/dd/yyyy' (to select all the dataset type 'all')", default='all')
parser.add_argument("--freq", type=str, help="Frequency of the interval selected, like '30min'", default='60min')
args = parser.parse_args()


def preprocessing_mov(df):
    df = df.rename(columns={'to_zone':'to_zone_fid', 'time':'to_timedate_gmt'})
    df = df.loc[(df.to_zone_fid > -1)].reset_index(drop=True)
    df_spark = df.to_spark()
    df_spark = df_spark.withColumn("to_timedate_gmt", expr("substring(to_timedate_gmt, 1, length(to_timedate_gmt)-3)"))
    df = df_spark.to_koalas()

    #print(df.head(5))

    return df


# Generator of calendar with specific timestamp (start/end='dd-mm-yyyy', freq='20min', '8hour', ...)
def calendar(start, end, freq):
    calendar = pd.date_range(start=start, end=end, freq=freq)
    return calendar


def features_matrix_mov(dataframe, min_date, max_date, intv):
    shapefile = gpd.read_file(pathShapeF)
    lista_zone = shapefile.FID.to_list()


    if min_date == 'all' and max_date == 'all':
        start = ks.to_datetime(dataframe.to_timedate_gmt.min(), format='%Y/%m/%d %H:%M')  # Lower date in the df
        min = start.replace(minute=00)  # Set minutes to 00
        end = ks.to_datetime(dataframe.to_timedate_gmt.max(), format='%Y/%m/%d %H:%M')  # Upper date in the df
        max = end.replace(hour=23, minute=59)  # Upper date set to 23:59
        cal = calendar(min, max, intv)
    else:
        cal = calendar(min_date, max_date, intv)

    cal = cal.to_list()
    cal = ks.to_datetime(cal)

    index_zone = list(range(len(lista_zone)))  # List from 0 to num zones
    map_zone = dict(list(zip(lista_zone, index_zone)))  # Map each element with its position number

    list_matrix_tot = []
    df_check = dataframe.to_spark().cache()
    lista_zone_2 = spark.createDataFrame(lista_zone, IntegerType()).select(col('value').alias('to_zone_fid')).cache()

    for j in range(0, len(cal) - 1):
        print('---FEAT---', j)
        # Dataset reduction with each time interval
        dataframe_ridotto = df_check.filter(((df_check.to_timedate_gmt >= cal[j]) &
                                      (df_check.to_timedate_gmt < cal[j + 1])))
        #print(dataframe_ridotto)

        #print('---SHAPEADJ---', j)

        df_spark = dataframe_ridotto
        df_grouped = df_spark.groupBy('to_zone_fid').mean('n_auto_park').withColumnRenamed('avg(n_auto_park)','media')
        df_grouped = lista_zone_2.join(df_grouped, on = ['to_zone_fid'], how='left').fillna(0).orderBy(col('to_zone_fid'))
        matrix = df_grouped.select('media').to_koalas().values.reshape((len(lista_zone), 1))
        matrix = csr_matrix(matrix)
        #print(matrix)
        list_matrix_tot += [matrix]
        del matrix

    df_check.unpersist()
    lista_zone_2.unpersist()
    #print(list_matrix_tot[0])
    return list_matrix_tot


if __name__ == '__main__':

        #.config('spark.driver.memory',"126g")\
    spark = SparkSession.builder\
        .master("local[10]")\
        .appName("Pyspark")\
        .config('spark.driver.memory', "60g")\
        .config('spark.ui.port', '4050')\
        .config("spark.local.dir", "/home/gpu2/spark-temp")\
        .config("spark.sql.session.timeZone', 'America/New_York")\
        .getOrCreate()
    spark.catalog.clearCache()
    
    
    # Upload CSV datas
    df = ks.read_csv(pathCSV, sep=';', dtype = {'time':str})
    #df = df.spark.cache()
    df = df.spark.repartition(100)
    df = preprocessing_mov(df)


    if args.mat == 'feat_mat':
        # Features Matrix
        result_fmat = features_matrix_mov(df, args.date_start, args.date_end, args.freq)
        with open(pathRes + f'features_matrix_sosta_media_contemporanea_{args.freq}.pickle', 'wb') as f:
            pickle.dump(result_fmat, f)
        print('File creato! (features_matrix_mov)')
        

    else:
        print('INCORRECT PARAMETERS!')
