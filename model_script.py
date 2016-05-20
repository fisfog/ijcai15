

import scipy as sp
import numpy as np
import xgboost as xgb
import math
import csv
import util
from sklearn import preprocessing


def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))


param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = 0.03
param['max_depth'] = 7
param['eval_metric'] = 'auc'
param['silent'] = 1
param['min_child_weight'] = 100
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['scale_pos_weight'] = 2
param['nthread'] = 4

 
pd={'colsample_bytree': [0.5,0.6,0.7,0.8,0.9],
 'eta': [0.06],
 'eval_metric': 'auc',
 'max_depth': [4],
 'min_child_weight': [100],
 'nthread': 4,
 'objective': 'binary:logitraw',
 'scale_pos_weight': 1,
 'silent': 1,
 'subsample': [0.9]}

train_data = sp.genfromtxt('train_feature_507.csv',delimiter=',',skip_header=1)
test_data = sp.genfromtxt('test_feature_506.csv',delimiter=',',skip_header=1)

is_nan=[63,64,65]

imp=preprocessing.Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imp.fit(train_data[:,is_nan])
train_data[:,is_nan]=imp.transform(train_data[:,is_nan])
for i in xrange(train_data.shape[0]):
	if train_data[:,is_nan[1]][i] == 2:
		train_data[:,is_nan[1]][i] = 0
	if train_data[:,is_nan[2]][i] == 0:
		train_data[:,is_nan[2]][i] = 3

imp=preprocessing.Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
imp.fit(test_data[:,is_nan])
test_data[:,is_nan]=imp.transform(test_data[:,is_nan])
for i in xrange(test_data.shape[0]):
	if test_data[:,is_nan[1]][i] == 2:
		test_data[:,is_nan[1]][i] = 0
	if test_data[:,is_nan[2]][i] == 0:
		test_data[:,is_nan[2]][i] = 3

# is_cat = np.array([62,63,64,65,94,95,180,191,202,213,224])
is_cat = np.array([62,63,64,65,94,95])
sc_index = range(2,train_data.shape[1]-1)
for i in is_cat:
	sc_index.remove(i)

scaler = preprocessing.StandardScaler().fit(train_data[:,sc_index])
train_data[:,sc_index] = scaler.transform(train_data[:,sc_index])
test_data[:,sc_index] = scaler.transform(test_data[:,sc_index])

# train_x = scaler.transform(train_data[:,sc_index])
# train_y = train_data[:,-1]
# param['scale_pos_weight'] = np.sum(train_y==0)*1.0/np.sum(train_y==1)
# test_x = scaler.transform(test_data[:,2:])

enc = preprocessing.OneHotEncoder(categorical_features=[59,60])
# enc=preprocessing.OneHotEncoder(n_values=[1672,8478,2,9,1672,8478,5000,5000,5000,5000,5000],categorical_features=is_cat-2)


dtrain = xgb.DMatrix(train_x,label=train_y)
dtest = xgb.DMatrix(test_x)
xgb.cv(param, dtrain, 450, nfold=5)

num_round = 430
bst = xgb.train(param, dtrain, num_round)
ypred=bst.predict(dtest)
result = dict()

for i in xrange(test_data.shape[0]):
	result[(str(int(test_data[i][0])),str(int(test_data[i][1])))]=sigmoid(ypred[i])

util.output_result(result,'gbdt_504_0.csv')


# jiang's feature process
fi = csv.reader(open('./512_new/3/t3_feature_f226_um_u_m_train.csv'))
htrain = fi.next()
label_index = htrain.index('label')
inx_train = [i for i in xrange(train_data.shape[1])]
inx_train.remove(0)
inx_train.remove(1)
inx_train.remove(label_index)

fi = csv.reader(open('./512_new/2/t3_feature_f202_um_u_m_test.csv'))
htest = fi.next()
label_index = htest.index('label')
inx_test = [i for i in xrange(test_data.shape[1])]
inx_test.remove(0)
inx_test.remove(1)
inx_test.remove(label_index)




