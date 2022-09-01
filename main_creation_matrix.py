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
from pyspark.sql.functions import expr, to_timestamp, col, unix_timestamp, first, sum, count, concat_ws, row_number, date_format,from_unixtime,lag, col, lead, to_date, unix_timestamp, datediff, lit, split
from pyspark.sql.types import *
from pyspark.sql.types import LongType, IntegerType
from pyspark.sql.window import Window
import timeit

os.chdir(Path(__file__).resolve().parent)

# Datapath file CSV (dataset)
pathCSV = 'data/trips2323_new.csv'

# Path ShapeFile
pathShapeF = 'data/shapefile/shapefile_2323.shp'

# Path for results
pathRes = 'results/'

# Skip division per 0
np.seterr(divide='ignore', invalid='ignore')

# Parameters to set type of matrix and value of timestamp
parser = argparse.ArgumentParser(description="Parameters to decide the matrix's type and date interval")
parser.add_argument("--mat", type=str, help="Select which matrix ('adj_mat', 'feat_mat', 'all')", default='all')
parser.add_argument("--date_start", type=str, help="Start date of the interval in the format 'mm/dd/yyyy' (to select all the dataset type 'all')", default='all')
parser.add_argument("--date_end", type=str, help="End date of the interval in the format 'mm/dd/yyyy' (to select all the dataset type 'all')", default='all')
parser.add_argument("--freq", type=str, help="Frequency of the interval selected, like '30min'")
args = parser.parse_args()


def preprocessing(df):
    #df_cache = df.spark.cache()
    df = df.rename(columns={'from_zone':'from_zone_fid', 'from_timedate':'from_timedate_gmt', 'to_zone':'to_zone_fid', 'to_timedate':'to_timedate_gmt'})

    df = df.dropna(subset=['to_zone_fid'])
    fill = {'from_zone_fid': -1}
    df = df.fillna(value=fill)


    # Initial Filter
    # (df.triptime_s < 3278)
    # & (df.stoptime_s < 123818)
    df = df.loc[ (df.tripdistance_m > 0) & (df.stoptime_s > 0)]

    #df = df.sort_values(by=['from_timedate_gmt', 'to_timedate_gmt'])
    #df = df.reset_index(drop = True)

    df_spark = df.to_spark()

    df_spark = df_spark.withColumn("from_timedate_gmt", expr("substring(from_timedate_gmt, 1, length(from_timedate_gmt)-3)"))
    df_spark = df_spark.withColumn("to_timedate_gmt", expr("substring(to_timedate_gmt, 1, length(to_timedate_gmt)-3)"))
    #df_spark = df_spark.withColumn("from_timedate_gmt", df_spark.from_timedate_gmt.substr(1, 6))

    df = df_spark.to_koalas()

    #df_cache.spark.unpersist()

    return df

# Fuction that count unique values of a specific column [col]
def count_zones(dataframe, col):
    lista_col = dataframe[col].to_list()
    result = []

    for value in dataframe[col].unique().to_list():
        result.append([value, lista_col.count(value)])

    return result


# Generator of calendar with specific timestamp (start/end='dd-mm-yyyy', freq='20min', '8hour', ...)
def calendar(start, end, freq):
    calendar = ks.date_range(start=start, end=end, freq=freq)
    return calendar

# Lower limit of times in the dataset
# Lower limit of times in the dataset
def time_limits_inf(df, intv):
    #df_cache = dataframe.spark.cache()

    # Generation of lower and upper bound of the calendar
    min = ks.to_datetime(df.from_timedate_gmt.min(), format='%Y/%m/%d %H:%M')  # Lower date in the df
    min = min.replace(minute=00)  # Set minutes to 00
    max = ks.to_datetime(df.to_timedate_gmt.max(), format='%Y/%m/%d %H:%M')  # Upper date in the df
    max = max.replace(hour=23, minute=59)  # Upper date set to 23:59

    # Create calendar
    cal = calendar(min, max, intv)
    cal = cal.to_frame(index=False, name='date')
    cal = cal.to_spark()
    
    #print(cal.show())
    df_spark = df.to_spark()
    
    df_spark.createOrReplaceTempView('df_spark')
    cal.createOrReplaceTempView('cal')
    res1 = df_spark.join(cal).where((to_date(df_spark.from_timedate_gmt) == to_date(cal.date)) & \
                               (df_spark.from_timedate_gmt.cast(TimestampType()) >= cal.date.cast(TimestampType()))).withColumnRenamed('date','date_1')#.crossJoin(cal).withColumnRenamed('date', 'date_2')#.crossJoin(cal).withColumnRenamed('date', 'date_3')

    res1 = res1.drop('from_timedate_gmt')#, 'to_timedate_gmt')#, 'date_2')
    res1 = res1.withColumnRenamed('date_1', 'from_timedate_gmt')
    w = Window.partitionBy("idtrajectory").orderBy(col("from_timedate_gmt").desc()) #, col("to_timedate_gmt").desc())
    res1 = res1.withColumn("row", row_number().over(w)).where("row == 1").drop("row")
    res1.createOrReplaceTempView('res1')
    
    res2 = res1.join(cal).where((to_date(df_spark.to_timedate_gmt) == to_date(cal.date)) & \
                               (df_spark.to_timedate_gmt.cast(TimestampType()) >= cal.date.cast(TimestampType()))).withColumnRenamed('date','date_1')
    
    res2 = res2.drop('to_timedate_gmt')#, 'to_timedate_gmt')#, 'date_2')
    res2 = res2.withColumnRenamed('date_1', 'to_timedate_gmt')
    w = Window.partitionBy("idtrajectory").orderBy(col("to_timedate_gmt").desc())
    res2 = res2.withColumn("row", row_number().over(w)).where("row == 1").drop("row")
    dataframe = res2.to_koalas()

    return dataframe


# Add 'weight' column calculated by making the quotient between the chosen
# interval and the delta of the time taken for the travel
def add_peso_column(dataframe, calendario):
    delta_cal = calendario.to_list()[1] - calendario.to_list()[0]
    delta_cal = delta_cal.seconds

    df_spark = dataframe.to_spark()
    df_spark = df_spark.withColumn('peso', col('to_timedate_gmt').cast(LongType()) - col('from_timedate_gmt').cast(LongType()))\
                       .withColumn('peso', delta_cal / (col('peso') + delta_cal))
    dataframe = df_spark.to_koalas()
    return dataframe


# Add 'totimedate_plus_stoptime' column calculated by making the sum of to_date and the stoptime
def add_new_stoptime_column(df):
    #df_cache = df.spark.cache()
    df_spark = df.to_spark()
    
    df_spark = df_spark.withColumn('new_stoptime', col('stoptime_s') + 60 - (col('stoptime_s') % 60))\
                       .withColumn('totimedate_plus_stoptime', (unix_timestamp(col('to_timedate_gmt').cast(TimestampType())) + col('new_stoptime')))\
                       .withColumn('totimedate_plus_stoptime', to_timestamp(col('totimedate_plus_stoptime')))

    df_spark = df_spark.drop('new_stoptime')

    df = df_spark.to_koalas()

    return df

# Creation of transition matrix
def transition_matrix(df, min_date, max_date, intv):
    shapefile = gpd.read_file(pathShapeF)
    lista_zone = shapefile.FID.to_list()
    #lista_zone.insert(0, -1)
    
    # Lower bound of time interval
    df = time_limits_inf(df, intv)

    index_zone = list(range(len(lista_zone)))           # List from 0 to -1 (zones)
    map_zone = dict(list(zip(lista_zone, index_zone)))  # Map each element with its position number

    if min_date == 'all' and max_date == 'all':
        # Take-over calendar bounds
        min = df.from_timedate_gmt.min()
        max = df.to_timedate_gmt.max()
        cal = calendar(min, max, intv)
    else:
        cal = calendar(min_date, max_date, intv)

    df = add_peso_column(df, cal)
    cal = cal.to_list()

    list_matrix_tot = []
    df_check = df.to_spark().cache()

    lista_zone_1 = spark.createDataFrame(lista_zone, IntegerType()).select(col('value').alias('from_zone_fid'))
    lista_zone_2 = spark.createDataFrame(lista_zone, IntegerType()).select(col('value').alias('to_zone_fid'))
    crossListe = lista_zone_1.crossJoin(lista_zone_2).cache()

    for j in range(0, len(cal) - 1):
        print('---ADJ---', j)
        start_time = timeit.default_timer()
        dataframe_ridotto = df_check.filter(((df_check.from_timedate_gmt >= cal[j]) &
                                  (df_check.from_timedate_gmt < cal[j + 1]))|
                                    ((df_check.to_timedate_gmt >= cal[j]) &
                                           (df_check.to_timedate_gmt <= cal[j + 1])) |
                                    ((df_check.from_timedate_gmt < cal[j]) &
                                           (df_check.to_timedate_gmt > cal[j + 1])))


       
        print('---SHAPEADJ---', j)

        df_spark = dataframe_ridotto #.to_spark()

        df_spark = df_spark.withColumn("peso", df_spark["peso"].cast(LongType()))
        df_grouped = df_spark.groupBy('from_zone_fid', 'to_zone_fid').agg(sum('peso').alias('peso'))

        df_grouped = crossListe.join(df_grouped, on = ['from_zone_fid', 'to_zone_fid'], how='left').fillna(0)

        matrix = df_grouped.select('peso').to_koalas().values.reshape((len(lista_zone), len(lista_zone)))


        #else:
        #   matrix = np.zeros((len(lista_zone), len(lista_zone)))
        #    matrix = csr_matrix(matrix)
        #    list_matrix_tot += [matrix]
        #    continue

        #matrix = [el / sum(el) for el in matrix]
        matrix = (matrix.T / matrix.sum(axis=1)).T
        matrix = np.nan_to_num(matrix)
        matrix = csr_matrix(matrix)
        list_matrix_tot += [matrix]
        stop_time = timeit.default_timer()
        print('Time: ', stop_time - start_time)
        del matrix

    df_check.unpersist()
    crossListe.unpersist()
    return list_matrix_tot

def features_matrix(dataframe, min_date, max_date, intv):
    shapefile = gpd.read_file(pathShapeF)
    lista_zone = shapefile.FID.to_list()
    #lista_zone.insert(0, -1)

    if min_date == 'all' and max_date == 'all':
        start = ks.to_datetime(dataframe.from_timedate_gmt.min(), format='%Y/%m/%d %H:%M')  # Lower date in the df
        min = start.replace(minute=00)  # Set minutes to 00
        end = ks.to_datetime(dataframe.to_timedate_gmt.max(), format='%Y/%m/%d %H:%M')  # Upper date in the df
        max = end.replace(hour=23, minute=59)  # Upper date set to 23:59
        # Take-over calendar bounds
        #min = dataframe.from_timedate_gmt.min()
        #max = dataframe.to_timedate_gmt.max()
        cal = calendar(min, max, intv)
    else:
        cal = calendar(min_date, max_date, intv)

    cal = cal.to_list()
    cal = ks.to_datetime(cal)
        # Add totimedate_plus_stoptime column
    dataframe = add_new_stoptime_column(dataframe)

    index_zone = list(range(len(lista_zone)))  # List from 0 to num zones
    map_zone = dict(list(zip(lista_zone, index_zone)))  # Map each element with its position number

    list_matrix_tot = []
    df_check = dataframe.to_spark().cache()
    lista_zone_2 = spark.createDataFrame(lista_zone, IntegerType()).select(col('value').alias('to_zone_fid')).cache()

    for j in range(0, len(cal) - 1):
        print('---FEAT---', j)
        # Dataset reduction with each time interval
        dataframe_ridotto = df_check.filter(((df_check.to_timedate_gmt >= cal[j]) &
                                      (df_check.to_timedate_gmt < cal[j + 1])) |
                                     ((df_check.totimedate_plus_stoptime > cal[j]) &
                                      (df_check.totimedate_plus_stoptime <= cal[j + 1])) |
                                     ((df_check.to_timedate_gmt < cal[j]) &
                                      (df_check.totimedate_plus_stoptime > cal[j + 1])))

        #if dataframe_ridotto.shape[0] != 0:
        print('---SHAPEADJ---', j)

        df_spark = dataframe_ridotto #.to_spark()
          #print(df_spark)
        df_grouped = df_spark.groupBy('to_zone_fid').agg(count('to_zone_fid').alias('count'))
        df_grouped = lista_zone_2.join(df_grouped, on = ['to_zone_fid'], how='left').fillna(0)
        matrix = df_grouped.select('count').to_koalas().values.reshape((len(lista_zone), 1))
        matrix = csr_matrix(matrix)
        list_matrix_tot += [matrix]
        del matrix

        #else:
        #  matrix = np.zeros((len(lista_zone), 1))
        #  matrix = csr_matrix(matrix)
        #  list_matrix_tot += [matrix]
        #  del matrix
        #  continue


    df_check.unpersist()
    lista_zone_2.unpersist()
    return list_matrix_tot

def stoptime_bound(df):
    #Upper Bound e Lower Bound
    print(df.stoptime_s.describe())
    Q1 = df.stoptime_s.describe().to_list()[4]
    Q2 = df.stoptime_s.describe().to_list()[5]
    Q3 = df.stoptime_s.describe().to_list()[6]
    IQR = Q3 - Q1
    L = Q1 - (1.5 * IQR)
    U = Q3 + (1.5 * IQR)
    print('Lower: ', L)
    print('Upper', U)



if __name__ == '__main__':
    #ks.set_option('compute.default_index_type', 'distributed') #-sequence')
    #ks.set_option('compute.ops_on_diff_frames', True)
    #ks.set_option('compute.shortcut_limit', 20000000)
    #ks.set_option('compute.max_rows', 20000000)
            #.config("spark.driver.memory",  "128g") \
    
    #spark = SparkSession.builder\
    #    .master("local")\
    #    .appName("Pyspark")\
    #    .config("spark.driver.memory",  "60g") \
    #    .config("spark.network.timeout", "300s")\
    #    .config("spark.memory.fraction", 0.6)\
    #    .config('executor.cores', 110)\
    #    .config("spark.sql.codegen.wholeStage", False)\
    #    .config("spark.sql.autoBroadcastJoinThreshold", "-1")\
    #    .config("spark.executor.heartbeatInterval", '299s')\
    #    .config("spark.executor.memory", '60g')\
    #    .config("spark.checkpoint.compress", True)\
    #    .getOrCreate()

    #spark.catalog.clearCache()
    #.config('spark.executor.cores', "100")\
    
    # Simple spark configuration
    spark = SparkSession.builder\
        .master("local[10]")\
        .appName("Pyspark")\
        .config('spark.driver.memory', "126g")\
        .config('spark.ui.port', '4050')\
        .config("spark.local.dir", "/home/gpu2/spark-temp")\
        .getOrCreate()
    spark.catalog.clearCache()
    
    
    # Upload CSV datas
    df = ks.read_csv(pathCSV, sep=';', dtype = {'from_timedate':str, 'to_timedate':str})
    #df = df.spark.cache()
    df = df.spark.repartition(10)
    df = preprocessing(df)

    print(df.head())
    exit()

    print('-----------OUTPUT------------')

    if args.mat == 'adj_mat':
        # ADJ Matrix
        result_tmat = transition_matrix(df, args.date_start, args.date_end, args.freq)
        with open(pathRes + f'transition_matrix_{args.freq}.pickle', 'wb') as f:
            pickle.dump(result_tmat, f)
        print('File creato!')

    elif args.mat == 'feat_mat':
        # Features Matrix
        result_fmat = features_matrix(df, args.date_start, args.date_end, args.freq)
        with open(pathRes + f'features_matrix_{args.freq}.pickle', 'wb') as f:
            pickle.dump(result_fmat, f)
        print('File creato!')

    elif args.mat == 'all':
        # ADJ Matrix
        result_tmat = transition_matrix(df, args.date_start, args.date_end, args.freq)
        with open(pathRes + f'transition_matrix_{args.freq}.pickle', 'wb') as f:
            pickle.dump(result_tmat, f)
        print('File creato!')
        # Features Matrix
        result_fmat = features_matrix(df, args.date_start, args.date_end, args.freq)
        with open(pathRes + f'features_matrix_{args.freq}.pickle', 'wb') as f:
            pickle.dump(result_fmat, f)
        print('File creato!')

    else:
        print('INCORRECT PARAMETERS!')