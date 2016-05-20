#-*- coding:utf-8 -*-
"""
stacking learning with pairwiseLR
"""

import lrEnsemble
import util
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold



homedir = './ensemble/train/'
train_label_file = 'train_format1.csv'
f_list = [homedir+d for d in os.listdir(homedir)]
label_dict = util.read_label(train_label_file)
combo_feature = lrEnsemble.combine_result(f_list)

train_data = np.array(combo_feature.values(),dtype=float64)
train_label = np.array([label_dict[um] for um in combo_feature.keys()],dtype=int)

kf = KFold(train_data.shape[0],n_folds=2)
for train_idx,test_idx in kf:
	cv_train_data = train_data[train_idx]
	cv_train_label = train_label[train_idx]
	cv_test_data = train_data[test_idx]
	cv_test_label = train_data[test_idx]
	pw_train_x, pw_train_y = lrEnsemble.pair_dataset(cv_train_data.tolist(),cv_train_label.tolist())
	lr = LogisticRegression(C=0.1)
	lr.fit(pw_train_x,pw_train_y)
	test_score = lrEnsemble.pwlr_predict(cv_train_data.tolist(),cv_train_label.tolist(),cv_test_data.tolist(),lr.coef_)
	print roc_auc_score(cv_test_label,test_score)

