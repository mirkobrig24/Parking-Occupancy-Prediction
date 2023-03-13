import pickle
import numpy as np
import csv
import pandas as pd
from scipy.sparse import csr_matrix



with open('results/transition_matrix_1H.pickle', 'rb') as f:
    x = pickle.load(f)
tot = x[0]
for i in range(1, len(x)):
    tot = tot + x[i]
tot = tot / len(x)

with open('results/MEAN_transition_matrix_1H.pickle', 'wb') as f:
    pickle.dump(tot, f)