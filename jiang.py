# -*- coding:utf-8 -*-
import sys
sys.path.append('D:/shaohua_jiang/________________IJCAI Competition/xgboost-master/wrapper')
import scipy as sp
import numpy as np
import xgboost as xgb
import math
import csv
import __util
import time
# import smote
# from sklearn import preprocessing
# from sklearn.cross_validation import KFold
print "---------------------------------------------"
print "--------------- Python Begin-----------------"
print "---------------------------------------------"
print time.localtime()

def sigmoid(x):
	return 1.0 / (1.0 + math.exp(-x))
feature_number = "277"
cur_p = [0.005,10,1300]
cur_p_s = "[" + str(cur_p[0]) + "," + str(cur_p[1]) + "," + str(cur_p[2])  + "]"
CV_FLAG = 0
# 输入输出文件
train_file =  "data/t3_feature_f"+feature_number+"_um_u_m_train.csv"
test_file =  "data/t3_feature_f"+feature_number+"_um_u_m_test.csv"
train_prob = "result/t3_feature_f"+feature_number+"_um_u_m_train_prob"+cur_p_s+".csv"
test_prob = "result/t3_feature_f"+feature_number+"_um_u_m_test_prob"+cur_p_s+".csv"
print train_file
print cur_p

# 参数设置
param = {}
param['objective'] = 'binary:logitraw'
param['eta'] = cur_p[0]
param['max_depth'] = cur_p[1]
param['eval_metric'] = 'auc'
param['silent'] = 1
param['min_child_weight'] = 100
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 4
param['scale_pos_weight'] = 2

train_data = sp.genfromtxt(train_file,delimiter=',',skip_header=1)


# 填充nan值为0
for i in xrange(train_data.shape[0]):
	for j in xrange(train_data.shape[1]):
		if np.isnan(train_data[i][j]):
			train_data[i][j] = 0


print "this is read train data func"
fi = csv.reader(open(train_file))
htrain = fi.next()
label_index = htrain.index('label')
inx_train = [i for i in xrange(train_data.shape[1])]
inx_train.remove(277)
inx_train.remove(0)
inx_train.remove(1)
inx_train.remove(label_index)
train_x = train_data[:,inx_train]
train_y = train_data[:,label_index]


# 转化成xgb DMatrix
dtrain = xgb.DMatrix(train_x, label=train_y)

if CV_FLAG == 1:
    print "this is cv func"
    # cv
    xgb.cv(param, dtrain,cur_p[2], nfold=5)

# train
num_round = cur_p[2]
print "this is Train Model func"
bst = xgb.train(param, dtrain, num_round)

test_data = sp.genfromtxt(test_file,delimiter=',',skip_header=1)
for i in xrange(test_data.shape[0]):
	for j in xrange(test_data.shape[1]):
		if np.isnan(test_data[i][j]):
			test_data[i][j] = 0

print "this is read test data func"
fi = csv.reader(open(test_file))
htest = fi.next()
label_index = htest.index('label')
inx_test = [i for i in xrange(test_data.shape[1])]
inx_test.remove(277)
inx_test.remove(0)
inx_test.remove(1)
inx_test.remove(label_index)
test_x = test_data[:,inx_test]
dtest = xgb.DMatrix(test_x)
# predict
print "this is Predict func"
train_ypred = bst.predict(dtrain)
test_ypred = bst.predict(dtest)

train_result = dict()
for i in xrange(train_data.shape[0]):
	train_result[(str(int(train_data[i][0])),str(int(train_data[i][1])))]=sigmoid(train_ypred[i])
__util.output_result(train_result,train_prob)

test_result = dict()
for i in xrange(test_data.shape[0]):
	test_result[(str(int(test_data[i][0])),str(int(test_data[i][1])))]=sigmoid(test_ypred[i])

__util.output_result(test_result,test_prob)

print time.localtime()
print "---------------------------------------------"
print "--------------- Python End-------------------"
print "---------------------------------------------"


