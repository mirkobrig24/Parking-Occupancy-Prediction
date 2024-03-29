import pickle
import numpy as np
import csv
import pandas as pd
from scipy.sparse import csr_matrix

'''
df = pd.read_csv('C:/Users/Mirko/Desktop/feat3.csv', header=None)
feat = df.to_numpy().T
print(feat.shape)
feat_resh = feat.reshape((1095, 54, 43))
print(feat_resh.shape)
'''
'''
feat = pd.read_csv('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Features Matrix/Senza -1/feat_1h.csv', header=None)
feat_np = feat.to_numpy()
feat_4month = feat_np[:, 5832:]
print(feat_4month.shape)
pd.DataFrame(feat_4month).to_csv('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Features Matrix/Senza -1/feat_4month_1h.csv', header=False, index=False)

feat = pd.read_csv('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Features Matrix/Senza -1/feat_1month_1h.csv', header=None)
print(feat.shape)
'''
'''
f = np.zeros((50, 4, 4))
lista = list()
for i in range(0, 30):
    lista.append(np.array(f[i : i + 12]))

print(np.array(lista).shape)



with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/Senza -1/t_mat_TOT_1h.pickle', 'rb') as f:
    x = pickle.load(f)
#2928
x_4month = x[5832:]
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/Senza -1/t_mat_4month_1h.pickle', 'wb') as f:
    pickle.dump(x_4month, f)

tot = x_4month[0]
for i in range(1, len(x_4month)):
    tot = tot + x_4month[i]

tot = tot / len(x_4month)

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/Senza -1/t_mat_MEAN_4month_1h.pickle', 'wb') as f:
    pickle.dump(tot, f)
#print(len(x))
exit()
'''

'''
new_adj = list()
for i in range(0, len(x)):
    tmp = x[i]
    new_adj.append(tmp[1:2323, 1:2323])

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/Senza -1/t_mat_TOT.pickle', 'wb') as f:
    pickle.dump(new_adj, f)

print(type(new_adj))
print(len(new_adj))
'''
'''
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Features Matrix/f_mat_TOT.pickle', 'rb') as f:
    x = pickle.load(f)

#x_0 = x[0].todense()
tot=np.zeros((2322, 1))
for i in range(1, 4380):
    temp = x[i].todense()
    tot = np.append(tot, temp, axis=1)
print(tot.shape)
pd.DataFrame(tot).to_csv('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Features Matrix/f_mat_6month.csv', header=False, index=False)
'''
'''
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_TOT.pickle', 'rb') as f:
    x = pickle.load(f)
reduce = x[:4380]
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_6month.pickle', 'wb') as f:
    pickle.dump(reduce, f)
#print(len(reduce))
'''



# CREAZIONE DEL CSV FEAT

with open('results/features_matrix_1H.pickle', 'rb') as f:
    x = pickle.load(f)

#x_0 = x[0].todense()
tot=np.zeros((2322, 1))
for i in range(1, len(x)):
    temp = x[i].todense()
    tot = np.append(tot, temp, axis=1)

print(tot.shape)
pd.DataFrame(tot).to_csv('results/feat_1h.csv', header=False, index=False)

'''
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Features Matrix/f_mat_TOT.pickle', 'rb') as f:
    x = pickle.load(f)

print(len(x))
with open('C:/Users/Mirko/Desktop/t_mat_MEAN.pickle', 'rb') as f:
    x = pickle.load(f)
adj = np.array(x.todense(), dtype=np.float32)
print(adj)
exit()
tot = x[0]
for i in range(1, len(x)):
    tot = tot + x[i]

tot = tot / len(x)
with open('C:/Users/Mirko/Desktop/t_mat_MEAN.pickle', 'wb') as f:
    pickle.dump(tot, f)
#print(len(x))
exit()
'''

'''
t = list()
x = np.array((2,2))
y = np.array([3,4,3])

t.append(x)
t.append(y)
t = np.array(t, dtype=object)
print(type(t))
'''
'''
tot = list()
a = np.zeros((2, 2))
b = np.zeros((2, 2))
tot.append(a)
tot.append(b)
print(np.array(tot).shape)

#test_X_feat.append(np.array(test_data_feat[i : i + seq_len]))
'''
'''
    adj_dense = []
    for i in range(len(adj_sparse)):
        #adj_dense += [adj_sparse[i].todense()]
        tmp = adj_sparse[i].todense()
        tmp = tmp[1:2323, 1:2323]
        tmp = csr_matrix(tmp)
        adj_dense.append(tmp)

with open('C:/Users/Mirko/Desktop/t_mat_TOT.pickle', 'wb') as f:
    pickle.dump(adj_dense, f)
'''
'''
adj = pd.read_csv('C:/Users/Mirko/Desktop/adj2.csv', header=None)
#df = df.iloc[1:2323, 1:2323]

#feat = pd.read_csv('C:/Users/Mirko/Downloads/T-GCN-master/T-GCN-master/T-GCN/T-GCN-PyTorch/data/los_speed.csv', header=None)
adj = np.array(adj)

print('Adj', adj[:, :4])
#print('Feat', feat.shape)

#feat.to_csv('C:/Users/Mirko/Desktop/feat2.csv', header=False, index=False)
#df.to_csv('C:/Users/Mirko/Desktop/adj2.csv', header=False, index=False)
'''
'''
with open('C:/Users/Mirko/Desktop/adj.csv') as f:
    x = csv.reader(f, delimiter=',')
    rows = []
    for row in x:
        rows.append(row)
x = np.array(rows)
print(x.shape)
'''
'''
with open('C:/Users/Mirko/Desktop/t_mat_TOT_1.pickle', 'rb') as f:
    x = pickle.load(f)

np.savetxt('C:/Users/Mirko/Desktop/foo.csv', x, delimiter=',')
'''

'''
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/8hour/Features Matrix/f_mat_TOT.pickle', 'rb') as f:
    x = pickle.load(f)

#x_0 = x[0].todense()
tot=np.zeros((2322, 1))
for i in range(1, len(x)):
    temp = x[i].todense()
    tot = np.append(tot, temp, axis=1)

print(tot.shape)
#tot.to_csv('C:/Users/Mirko/Desktop/feat3.csv', header=False, index=False)
print(tot.T.shape)
#np.savetxt('C:/Users/Mirko/Desktop/feat3.csv', tot, delimiter=',')

#with open('C:/Users/Mirko/Desktop/t_mat_TOT_1.pickle', 'wb') as f:
    #pickle.dump(x_0, f)
'''

'''
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/8hour/Transitions Matrix/t_mat_january_febr.pickle', 'rb') as f:
    x1 = pickle.load(f
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/8hour/Transitions Matrix/t_mat_march_april_may.pickle', 'rb') as f:
    x2 = pickle.load(f)

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/8hour/Transitions Matrix/t_mat_june_july_august.pickle', 'rb') as f:
    x3 = pickle.load(f)
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/8hour/Transitions Matrix/t_mat_sept_oct_nov_dec.pickle', 'rb') as f:
    x4 = pickle.load(f)
with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_april_may.pickle', 'rb') as f:
    x5 = pickle.load(f)

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_31May_31June.pickle', 'rb') as f:
    x6 = pickle.load(f)

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_july_august.pickle', 'rb') as f:
    x7 = pickle.load(f)

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_sept_oct.pickle', 'rb') as f:
    x8 = pickle.load(f)

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/60min/Transitions Matrix/t_mat_nov_dec.pickle', 'rb') as f:
    x9 = pickle.load(f)

xtot = []

xtot = xtot + x1
xtot = xtot + x2
xtot = xtot + x3
xtot = xtot + x4
xtot = xtot + x5
xtot = xtot + x6
xtot = xtot + x7
xtot = xtot + x8
xtot = xtot + x9

with open('C:/Users/Mirko/Documents/UNIMIB/TESI/script_tesi/Matrici/8hour/Transitions Matrix/t_mat_TOT.pickle', 'wb') as f:
    pickle.dump(xtot, f)

print(len(xtot))


pickle_file = open("C:/Users/Mirko/Desktop/transition_matrix.pickle", "rb")
objects = []
while True:
    try:
        objects += [pickle.load(pickle_file)]z
    except EOFError:
        break
pickle_file.close()

print(len(objects))
'''