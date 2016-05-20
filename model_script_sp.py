# -*- coding:utf-8 -*-

import scipy as sp
import numpy as np
import xgboost as xgb
import math
import csv
import util
# import smote
# from sklearn import preprocessing
# from sklearn.cross_validation import KFold

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))

# 输入输出文件
train_file = ''
test_file = ''
train_prob = ''
test_prob = ''

# 参数设置
param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = 0.02
param['max_depth'] = 8
param['eval_metric'] = 'auc'
param['silent'] = 1
param['min_child_weight'] = 100
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 4
param['scale_pos_weight'] = 2

train_data = sp.genfromtxt(train_file,delimiter=',',skip_header=1)
test_data = sp.genfromtxt(test_file,delimiter=',',skip_header=1)

# 填充nan值为0
for i in xrange(train_data.shape[0]):
	for j in xrange(train_data.shape[1]):
		if np.isnan(train_data[i][j]):
			train_data[i][j] = 0

for i in xrange(test_data.shape[0]):
	for j in xrange(test_data.shape[1]):
		if np.isnan(test_data[i][j]):
			test_data[i][j] = 0


fi = csv.reader(open(train_file))
htrain = fi.next()
label_index = htrain.index('label')
inx_train = [i for i in xrange(train_data.shape[1])]
inx_train.remove(0)
inx_train.remove(1)
inx_train.remove(label_index)
train_x = train_data[:,inx_train]
train_y = train_data[:,label_index]

fi = csv.reader(open(test_file))
htest = fi.next()
label_index = htest.index('label')
inx_test = [i for i in xrange(test_data.shape[1])]
inx_test.remove(0)
inx_test.remove(1)
inx_test.remove(label_index)
test_x = test_data[:,inx_test]

# 转化成xgb DMatrix
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)

# cv
xgb.cv(param, dtrain, 450, nfold=5)

# train
num_round = 360
bst = xgb.train(param, dtrain, num_round)

# predict
train_ypred = bst.predict(dtrain)
test_ypred = bst.predict(dtest)

train_result = dict()
for i in xrange(train_data.shape[0]):
	train_result[(str(int(train_data[i][0])),str(int(train_data[i][1])))]=sigmoid(train_ypred[i])
util.output_result(train_result,train_prob)

test_result = dict()
for i in xrange(test_data.shape[0]):
	test_result[(str(int(test_data[i][0])),str(int(test_data[i][1])))]=sigmoid(test_ypred[i])

util.output_result(test_result,test_prob)

