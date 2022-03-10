from __future__ import print_function
import os
from pickle import FALSE
import pandas as pd
import numpy as np
from datetime import datetime

from . import load_stdata
from ..utils import string2timestamp as s2t

def my_s2t(strings, T=3):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, hour = int(t[:4]), int(t[5:7]), int(t[8:10]), int(t[11:13])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour, minute=00)))

    return timestamps

class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=3, CheckComplete=FALSE, Hours0_23=True):
        super(STMatrix, self).__init__()
        print('DATA', data.shape)
        print('TIME', len(timestamps))
        #exit()
        data = np.transpose(data, (0, 3, 1, 2))
        print('DATA2', data.shape)
        #exit()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        string2timestamp = my_s2t if Hours0_23 else s2t
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=12, len_trend=0, TrendInterval=0, len_period=0, PeriodInterval=0):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        #offset_frame = pd.DateOffset(minutes=3 * 60 * 8 // self.T)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness+1),
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend+1)]]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y


if __name__ == '__main__':
    pass
