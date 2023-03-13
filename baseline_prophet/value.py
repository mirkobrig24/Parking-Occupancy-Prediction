import os
import pandas as pd


def read_csv_files(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    data = []
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        data.append(df)
    stats(pd.concat(data))

def stats(df):
    print('valore medio', df.mean())


read_csv_files('.')