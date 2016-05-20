#-*- coding:utf-8 -*-
"""
pairwiseLR
"""

import util
import numpy as np

def sigmoid(x):
	r = x
	if x > 19:
		r = 19
	if x < -19:
		r = -19
	return 1.0 / (1 + np.exp(r))

def be_positive(x):
	return sigmoid(-x)

def be_negative(x):
	return 1.0-be_positive(x)

def combine_result(feature_list):
	combo_feature = dict()
	for file_name in feature_list:
		for e in util.parse(file_name):
			user = e['user_id']
			merchant = e['merchant_id']
			prob = e['prob']
			um = (user,merchant)
			if um not in combo_feature:
				combo_feature[um] = []
			combo_feature[um].append(prob)
	return combo_feature

def pair_dataset(data,label):
	L = len(data)
	assert(L == len(label))
	pair_data = []
	pair_label = []
	for i in xrange(0,L-1):
		for j in xrange(i+1,L):
			pair_data.append(data[i]+data[j])
			lb = 0 if label[i]==label[j] else 1
			pair_label.append(lb)

	return np.array(pair_data),np.array(pair_label)

def pwlr_predict(train_data,train_label,test_data,weights):
	L = len(train_data)
	result = np.zeros(len(test_data))
	for i in xrange(len(test_data)):
		test_instance = test_data[i]
		for j in xrange(L):
			train_instance = train_data[j]
			sample_instance = test_instance + train_instance
			if train_label[j] == 1:
				result[i] += be_negative(np.dot(sample_instance,weights))
			else:
				result[i] += be_positive(np.dot(sample_instance,weights))
	return result/L


