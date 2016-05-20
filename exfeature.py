#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
	特征提取
"""
# import gzip
import numpy as np
from datetime import *
from sklearn import preprocessing
import util
import csv

def ex_feature_v1(um_file,user_beh_log,is_train,user_profile_file,out_feature):
	print "<user,merchant> pair: ",
	um_set = set()
	if is_train:
		um_label = dict()
		for e in util.parse(um_file):
			um_label[(e['user_id'],e['merchant_id'])] = e['label']
			um_set.add((e['user_id'],e['merchant_id']))
	else:
		for e in util.parse(um_file):
			um_set.add((e['user_id'],e['merchant_id']))
	print len(um_set)

	user_subset = set([um[0] for um in um_set])
	print "user: ",
	user_profile = dict()
	for e in util.parse(user_profile_file):
		user = e['user_id']
		if user in user_subset:
			user_profile[user] = [e[it] for it in e if it != 'user_id']
	print len(user_profile)

	merchant_subset = set([um[1] for um in um_set])
	print "merchant: %d"%(len(merchant_subset))

	double11 = date(2014,11,11)

	print "extract user feature"
	user_feature = dict()
	user_stat = dict()
	user_count = dict()
	user_dod = dict()
	n_user_feature = 29 + 3
	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		item = e['item_id']
		cat = e['cat_id']
		brand = e['brand_id']
		dat = util.time_proc(e['time_stamp'])
		if dat > date(2014,11,11):
			dat = date(2014,11,11)
		act = int(e['action_type'])
		if user in user_subset:
			if user not in user_feature:
				user_feature[user] = [0 for i in xrange(n_user_feature)]
				user_count[user] = [0 for i in xrange(4)]
				user_stat[user] = [set() for i in xrange(14)]
				user_dod[user] = [dict() for i in xrange(2)]
			user_count[user][act] += 1
			user_stat[user][act].add(merchant)
			user_stat[user][act+4].add(dat)
			if dat == double11:
				user_stat[user][act+8].add(merchant)
			if act == 2:
				user_stat[user][12].add((merchant,dat))
				if cat not in user_dod[user][0]:
					user_dod[user][0][cat] = 0
				user_dod[user][0][cat] += 1
				if brand not in user_dod[user][1]:
					user_dod[user][1][brand] = 0
				user_dod[user][1][brand] += 1
			user_stat[user][13].add((merchant,dat))


	for user in user_subset:
		for i in xrange(4):
			user_feature[user][i] = len(user_stat[user][i])
			user_feature[user][i+4] = len(user_stat[user][i+4])
			user_feature[user][i+8] = len(user_stat[user][i+8])
			user_feature[user][i+12] = (user_feature[user][i+8]*1.0) / user_feature[user][i] if user_feature[user][i] != 0 else 0
			if user_feature[user][i+4] > 1:
				last_day = np.sort(np.array(list(user_stat[user][i+4])))[-2]
				diff = (date(2014,11,11) - last_day).days
				user_feature[user][i+16] = 1.0/(diff+1)
		user_feature[user][20] = (user_count[user][2]*1.0) / user_count[user][0] if user_count[user][0] != 0 else 0
		user_feature[user][21] = (user_count[user][2]*1.0) / user_count[user][1] if user_count[user][1] != 0 else 0
		user_feature[user][22] = (user_count[user][2]*1.0) / user_count[user][3] if user_count[user][3] != 0 else 0
		user_feature[user][23] = (user_count[user][0]*1.0) / user_feature[user][4] if user_feature[user][4] != 0 else 0
		user_feature[user][24] = (user_count[user][2]*1.0) / user_feature[user][6] if user_feature[user][6] != 0 else 0
		user_m = dict()
		for md in user_stat[user][12]:
			if md[0] not in user_m:
				user_m[md[0]] = 0
			user_m[md[0]] += 1
		for m in user_m:
			if user_m[m] > 1:
				user_feature[user][25] += 1
		# 505 f29
		user_m = dict()
		for md in user_stat[user][13]:
			if md[0] not in user_m:
				user_m[md[0]] = 0
			user_m[md[0]] += 1
		for m in user_m:
			if user_m[m] >= 3:
				user_feature[user][29] += 1
		# 501 ~26
		user_feature[user][26] = (user_feature[user][0]*1.0) / user_count[user][0] if user_count[user][0] != 0 else 0
		user_feature[user][27] = (user_feature[user][2]*1.0) / user_count[user][2] if user_count[user][2] != 0 else 0
		# 505 ~31
		max_c = -1
		for c in user_dod[user][0]:
			if user_dod[user][0][c] > max_c:
				max_buy_cat = c
				max_c = user_dod[user][0][c]
		user_feature[user][30] = max_buy_cat

		max_b = -1
		for b in user_dod[user][1]:
			if user_dod[user][1][b] > max_b:
				max_buy_brand = b
				max_b = user_dod[user][1][b]
		user_feature[user][31] = max_buy_brand
	
	# 502 ~27
	user_buy_rank = np.argsort(np.array(map(lambda x:x[2],user_count.values()))).tolist()
	user_list = user_count.keys()
	for i in xrange(len(user_list)):
		user_feature[user_list[i]][28] = user_buy_rank.index(i)

	del user_stat
	del user_dod
	del user_count

	print "extract merchant feature"
	merchant_feature = dict()
	merchant_stat = dict()
	merchant_count = dict()
	merchant_dod = dict()
	n_merchant_feature = 27 + 3
	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		item = e['item_id']
		cat = e['cat_id']
		brand = e['brand_id']
		dat = util.time_proc(e['time_stamp'])
		if dat > date(2014,11,11):
			dat = date(2014,11,11)
		act = int(e['action_type'])
		if merchant in merchant_subset:
			if merchant not in merchant_feature:
				merchant_feature[merchant] = [0 for i in xrange(n_merchant_feature)]
				merchant_count[merchant] = [0 for i in xrange(4)]
				merchant_stat[merchant] = [set() for i in xrange(16)]
				merchant_dod[merchant] = [dict() for i in xrange(2)]
			merchant_stat[merchant][act].add(user)
			merchant_count[merchant][act] += 1
			if dat == double11:
				merchant_stat[merchant][act+4].add(user)
			if act == 2:
				merchant_stat[merchant][8].add(item)
				merchant_stat[merchant][9].add(brand)
				merchant_stat[merchant][10].add((user,dat))
			# 505
			if cat not in merchant_dod[merchant][0]:
				merchant_dod[merchant][0][cat] = 0
			merchant_dod[merchant][0][cat] += 1
			if brand not in merchant_dod[merchant][1]:
				merchant_dod[merchant][1][brand] = 0
			merchant_dod[merchant][1][brand] += 1

			merchant_stat[merchant][act+11].add(dat)
			merchant_stat[merchant][15].add((user,dat))


	for merchant in merchant_subset:
		for i in xrange(4):
			merchant_feature[merchant][i] = len(merchant_stat[merchant][i])
			merchant_feature[merchant][i+4] = len(merchant_stat[merchant][i+4])
			merchant_feature[merchant][i+8] = (merchant_feature[merchant][i+4]*1.0) / merchant_feature[merchant][i] if merchant_feature[merchant][i] != 0 else 0
			# 501
			merchant_feature[merchant][i+15] = len(merchant_stat[merchant][i+11])
		merchant_feature[merchant][12] = len(merchant_stat[merchant][8])
		merchant_feature[merchant][13] = len(merchant_stat[merchant][9])
		m_user = dict()
		for ud in merchant_stat[merchant][10]:
			if ud[0] not in m_user:
				m_user[ud[0]] = 0
			m_user[ud[0]] += 1
		for u in m_user:
			if m_user[u] > 1:
				merchant_feature[merchant][14] += 1
		# 505 f27
		m_user = dict()
		for ud in merchant_stat[merchant][15]:
			if ud[0] not in m_user:
				m_user[ud[0]] = 0
			m_user[ud[0]] += 1
		for u in m_user:
			if m_user[u] >= 5:
				merchant_feature[merchant][27] += 1
		# 501
		merchant_feature[merchant][19] = (merchant_feature[merchant][2]*1.0) / merchant_feature[merchant][0] if merchant_feature[merchant][0] != 0 else 0
		merchant_feature[merchant][20] = (merchant_feature[merchant][2]*1.0) / merchant_feature[merchant][1] if merchant_feature[merchant][1] != 0 else 0
		merchant_feature[merchant][21] = (merchant_feature[merchant][2]*1.0) / merchant_feature[merchant][3] if merchant_feature[merchant][3] != 0 else 0
		merchant_feature[merchant][22] = (merchant_feature[merchant][0]*1.0) / merchant_feature[merchant][15] if merchant_feature[merchant][15] != 0 else 0
		merchant_feature[merchant][23] = (merchant_feature[merchant][2]*1.0) / merchant_feature[merchant][17] if merchant_feature[merchant][17] != 0 else 0

		# 502 ~26
		merchant_feature[merchant][24] = (merchant_count[merchant][0]*1.0) / merchant_feature[merchant][0] if merchant_feature[merchant][0] != 0 else 0
		merchant_feature[merchant][25] = (merchant_count[merchant][2]*1.0) / merchant_feature[merchant][2] if merchant_feature[merchant][2] != 0 else 0

		# 505 ~29
		max_c = -1
		for c in merchant_dod[merchant][0]:
			if merchant_dod[merchant][0][c] > max_c:
				max_cat = c
				max_c = merchant_dod[merchant][0][c]
		merchant_feature[merchant][28] = max_cat
		max_d = -1
		for d in merchant_dod[merchant][1]:
			if merchant_dod[merchant][1][d] > max_d:
				max_brand = d
				max_d = merchant_dod[merchant][1][d]
		merchant_feature[merchant][29] = max_brand

	merchant_buy_rank = np.argsort(np.array(map(lambda x:x[2],merchant_count.values()))).tolist()
	merchant_list = merchant_count.keys()
	for i in xrange(len(merchant_list)):
		merchant_feature[merchant_list[i]][26] = merchant_buy_rank.index(i)
		

	del merchant_stat


	print "extract <user,merchant> behavior feature"
	
	um_beh_feature = dict()
	um_beh_stat = dict()
	n_um_feature = 20+6+2+2
	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		item = e['item_id']
		dat = util.time_proc(e['time_stamp'])
		if dat > date(2014,11,11):
			dat = date(2014,11,11)
		act = int(e['action_type'])
		um = (user,merchant)
		if um in um_set:
			if um not in um_beh_feature:
				um_beh_feature[um] = [0 for i in xrange(n_um_feature)]
				um_beh_stat[um] = [set() for i in xrange(12)]
			um_beh_feature[um][act] += 1
			um_beh_stat[um][act].add(dat)
			um_beh_stat[um][act+4].add(item)
			if dat == double11:
				um_beh_stat[um][act+8].add(item)

	for um in um_set:
		for i in xrange(4):
			um_beh_feature[um][4+i] = len(um_beh_stat[um][i])
			um_beh_feature[um][8+i] = len(um_beh_stat[um][i+4])
			um_beh_feature[um][12+i] = len(um_beh_stat[um][i+8])
			um_beh_feature[um][16+i] = (um_beh_feature[um][12+i]*1.0)/um_beh_feature[um][8+i] if um_beh_feature[um][8+i] != 0 else 0 
			if um_beh_feature[um][4+i] > 1:
				last_day = np.sort(np.array(list(um_beh_stat[um][i])))[-2]
				diff = (date(2014,11,11) - last_day).days
				um_beh_feature[um][20+i] = 1.0 / (diff+1)
		um_beh_feature[um][24] = (um_beh_feature[um][0]*1.0)/um_beh_feature[um][4] if um_beh_feature[um][4] != 0 else 0
		um_beh_feature[um][25] = (um_beh_feature[um][2]*1.0)/um_beh_feature[um][6] if um_beh_feature[um][6] != 0 else 0
		um_beh_feature[um][26] = (um_beh_feature[um][8]*1.0)/um_beh_feature[um][0] if um_beh_feature[um][0] != 0 else 0
		um_beh_feature[um][27] = (um_beh_feature[um][10]*1.0)/um_beh_feature[um][2] if um_beh_feature[um][2] != 0 else 0
		# 502 ~29
		um_beh_feature[um][28] = (user_feature[um[0]][23]*1.0)/merchant_feature[um[1]][25] if merchant_feature[um[1]][25] != 0 else 0
		um_beh_feature[um][29] = (user_feature[um[0]][24]*1.0)/merchant_feature[um[1]][26] if merchant_feature[um[1]][26] != 0 else 0
		


	del um_beh_stat

	

	print 'write to file'
	f = open(out_feature,'w')
	f.write('user_id,merchant_id,')
	for i in xrange(n_um_feature):
		f.write('umf'+str(i)+',')
	for i in xrange(n_user_feature):
		f.write('uf'+str(i)+',')
	f.write('age,gender,')
	for i in xrange(n_merchant_feature):
		if i != n_merchant_feature-1:
			f.write('mf'+str(i)+',')
		else:
			f.write('mf'+str(i))
	if is_train:
		f.write(',label\n')
	else:
		f.write('\n')

	for um in um_set:
		f.write(um[0]+','+um[1]+',')
		for feat in um_beh_feature[um]:
			f.write(str(feat)+',')
		for feat in user_feature[um[0]]:
			f.write(str(feat)+',')
		f.write(user_profile[um[0]][0]+','+user_profile[um[0]][1]+',')
		for i in xrange(n_merchant_feature):
			if i != n_merchant_feature-1:
				f.write(str(merchant_feature[um[1]][i])+',')
			else:
				f.write(str(merchant_feature[um[1]][i]))
		if is_train:
			f.write(','+um_label[um]+'\n')
		else:
			f.write('\n')
	f.close()


def ex_feature_v2(um_file,user_beh_log,sim_file,is_train,old_feature,out_feature):
	print "<user,merchant> pair: ",
	um_set = set()
	if is_train:
		um_label = dict()
		for e in util.parse(um_file):
			um_label[(e['user_id'],e['merchant_id'])] = e['label']
			um_set.add((e['user_id'],e['merchant_id']))
	else:
		for e in util.parse(um_file):
			um_set.add((e['user_id'],e['merchant_id']))
	print len(um_set)

	double11 = date(2014,11,11)

	print "load merchant similar merchant"
	merchant_sim_m = dict()
	for e in util.parse(sim_file):
		merchant_sim_m[e['merchant_id']] = [e['sim_m1'],e['sim_m2'],e['sim_m3'],e['sim_m4'],e['sim_m5']]


	print "new <user,merchant> pair: ",
	um_new_feature = dict()
	um_new_stat = dict()
	um_new_count = dict()
	n_new_feature = 10
	for um in um_set:
		user = um[0]
		for m in merchant_sim_m[um[1]]:
			if (user,m) not in um_new_feature:
				um_new_feature[(user,m)] = [0 for k in xrange(n_new_feature)]
				um_new_count[(user,m)] = [0 for k in xrange(4)]
				um_new_stat[(user,m)] = [set() for i in xrange(12)]
	print "%d"%(len(um_new_feature))

	print "extract new feature"

	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		item = e['item_id']
		dat = util.time_proc(e['time_stamp'])
		if dat > date(2014,11,11):
			dat = date(2014,11,11)
		act = int(e['action_type'])
		um = (user,merchant)
		if um in um_new_feature:
			um_new_count[um][act] += 1
			um_new_stat[um][act].add(dat)
			if dat == double11:
				um_new_stat[um][act+4].add(item)

	for um in um_new_feature:
		for i in xrange(4):
			um_new_feature[um][i] = len(um_new_stat[um][i])
			um_new_feature[um][4+i] = len(um_new_stat[um][i+4])
		um_new_feature[um][8] = (um_new_count[um][0]*1.0)/um_new_feature[um][0] if um_new_feature[um][0] != 0 else 0
		um_new_feature[um][9] = (um_new_count[um][2]*1.0)/um_new_feature[um][2] if um_new_feature[um][2] != 0 else 0
	
	del um_new_stat
	del um_new_count

	# new_f_index_in_oldf = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,25,26,27]

	print "add new feature to file"
	old_f = csv.reader(open(old_feature))
	old_header = old_f.next()[:-1]
	fo = open(out_feature,'w')
	fo.write('user_id,merchant_id,')
	for it in old_header[2:]:
		fo.write(it+',')

	if is_train:
		for i in xrange(n_new_feature*5):
			fo.write('nsf'+str(i)+',')
		fo.write('label\n')
	else:
		for i in xrange(n_new_feature*5):
			if i%10 == 0:
				fo.write('sim'+str(i/10)+',')
			if i != n_new_feature*5-1:
				fo.write('nsf'+str(i)+',')
			else:
				fo.write('nsf'+str(i))
		fo.write('\n')

	for l in old_f:
		fo.write(l[0]+','+l[1]+',')
		if is_train:
			for feat in l[2:-1]:
				fo.write(feat+',')
		else:
			for feat in l[2:]:
				fo.write(feat+',')
		for sim_m in merchant_sim_m[l[1]]:
			fo.write(sim_m+',')
			s_um = (l[0],sim_m)
			for i in xrange(n_new_feature):
				if sim_m == merchant_sim_m[l[1]][-1] and i == n_new_feature-1:
					fo.write(str(um_new_feature[s_um][i]))
				else:
					fo.write(str(um_new_feature[s_um][i])+',')
		if is_train:
			fo.write(','+um_label[(l[0],l[1])]+'\n')
		else:
			fo.write('\n')
	fo.close()
	

def ex_feature_v3(um_file,user_beh_log,is_train,old_feature,out_feature):
	print "<user,merchant> pair: ",
	um_set = set()
	if is_train:
		um_label = dict()
		for e in util.parse(um_file):
			um_label[(e['user_id'],e['merchant_id'])] = e['label']
			um_set.add((e['user_id'],e['merchant_id']))
	else:
		for e in util.parse(um_file):
			um_set.add((e['user_id'],e['merchant_id']))
	print len(um_set)

	user_subset = set([um[0] for um in um_set])
	print "user: %d"%(len(user_subset))

	merchant_subset = set([um[1] for um in um_set])
	print "merchant: %d"%(len(merchant_subset))

	double11 = date(2014,11,11)
	date_span1 = (date(2014,5,11),date(2014,9,30))
	date_span2 = (date(2014,10,1),date(2014,10,31))
	date_span3 = (date(2014,11,1),date(2014,11,10))

	total_days = (date(2014,11,11)-date(2014,5,11)).days
	span1_days = (date(2014,9,30)-date(2014,5,11)).days
	span2_days = (date(2014,10,31)-date(2014,10,1)).days
	span3_days = (date(2014,11,10)-date(2014,11,1)).days

	um_new_feature = dict()
	# um_new_stat = dict()
	um_new_count = dict()
	n_new_um_feature = 24 + 2

	user_new_feature = dict()
	# user_new_count = dict()
	user_new_stat = dict()
	n_new_user_feature = 36 + 2

	merchant_new_feature = dict()
	# merchant_new_count = dict()
	merchant_new_stat = dict()
	n_new_merchant_feature = 24
	
	print "extract new feature"
	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		item = e['item_id']
		cat = e['cat_id']
		brand = e['brand_id']
		dat = util.time_proc(e['time_stamp'])
		if dat > date(2014,11,11):
			dat = date(2014,11,11)
		act = int(e['action_type'])
		um = (user,merchant)
		if um in um_set:
			if um not in um_new_feature:
				um_new_feature[um] = [0 for i in xrange(n_new_um_feature)]
				um_new_count[um] = [0 for i in xrange(4)]
				# um_new_stat[um] = [set() for i in xrange(4)]
			if dat >= date_span1[0] and dat <= date_span1[1]:
				um_new_feature[um][act] += 1
			if dat >= date_span2[0] and dat <= date_span2[1]:
				um_new_feature[um][act+4] += 1
			if dat >= date_span3[0] and dat <= date_span3[1]:
				um_new_feature[um][act+8] += 1
			um_new_count[um][act] += 1
		# ex user feature
		if user in user_subset:
			if user not in user_new_feature:
				user_new_feature[user] = [0 for i in xrange(n_new_user_feature)]
				user_new_stat[user] = [set() for i in xrange(18)]
			user_new_stat[user][act].add(dat)
			if dat >= date_span1[0] and dat <= date_span1[1]:
				user_new_stat[user][act+4].add(dat)
			if dat >= date_span2[0] and dat <= date_span2[1]:
				user_new_stat[user][act+8].add(dat)
			if dat >= date_span3[0] and dat <= date_span3[1]:
				user_new_stat[user][act+12].add(dat)
			if act == 2:
				user_new_stat[user][16].add(cat)
				user_new_stat[user][17].add(brand)

		# ex merchant feature
		if merchant in merchant_subset:
			if merchant not in merchant_new_feature:
				merchant_new_feature[merchant] = [0 for i in xrange(n_new_merchant_feature)]
				merchant_new_stat[merchant] = [set() for i in xrange(18)]
			merchant_new_stat[merchant][act].add(user)
			if dat >= date_span1[0] and dat <= date_span1[1]:
				merchant_new_stat[merchant][act+4].add(user)
			if dat >= date_span2[0] and dat <= date_span2[1]:
				merchant_new_stat[merchant][act+8].add(user)
			if dat >= date_span3[0] and dat <= date_span3[1]:
				merchant_new_stat[merchant][act+12].add(user)
			merchant_new_stat[merchant][16].add(cat)
			merchant_new_stat[merchant][17].add(brand)

	for user in user_subset:
		for i in xrange(4):
			user_new_feature[user][i] = len(user_new_stat[user][i+4])
			user_new_feature[user][i+4] = len(user_new_stat[user][i+8])
			user_new_feature[user][i+8] = len(user_new_stat[user][i+12])
			user_new_feature[user][i+12] = (user_new_feature[user][i]*1.0)/len(user_new_stat[user][i]) if len(user_new_stat[user][i]) != 0 else 0
			user_new_feature[user][i+16] = (user_new_feature[user][i+4]*1.0)/len(user_new_stat[user][i]) if len(user_new_stat[user][i]) != 0 else 0
			user_new_feature[user][i+20] = (user_new_feature[user][i+8]*1.0)/len(user_new_stat[user][i]) if len(user_new_stat[user][i]) != 0 else 0
			user_new_feature[user][i+24] = (user_new_feature[user][i]*1.0)/span1_days
			user_new_feature[user][i+28] = (user_new_feature[user][i+4]*1.0)/span2_days
			user_new_feature[user][i+32] = (user_new_feature[user][i+8]*1.0)/span3_days
		user_new_feature[user][36] = len(user_new_stat[user][16])
		user_new_feature[user][37] = len(user_new_stat[user][17])

			
	for merchant in merchant_subset:
		for i in xrange(4):
			merchant_new_feature[merchant][i] = len(merchant_new_stat[merchant][i+4])
			merchant_new_feature[merchant][i+4] = len(merchant_new_stat[merchant][i+8])
			merchant_new_feature[merchant][i+8] = len(merchant_new_stat[merchant][i+12])
			merchant_new_feature[merchant][i+12] = (merchant_new_feature[merchant][i]*1.0)/len(merchant_new_stat[merchant][i]) if len(merchant_new_stat[merchant][i]) != 0 else 0
			merchant_new_feature[merchant][i+16] = (merchant_new_feature[merchant][i+4]*1.0)/len(merchant_new_stat[merchant][i]) if len(merchant_new_stat[merchant][i]) != 0 else 0
			merchant_new_feature[merchant][i+20] = (merchant_new_feature[merchant][i+8]*1.0)/len(merchant_new_stat[merchant][i]) if len(merchant_new_stat[merchant][i]) != 0 else 0

	for um in um_set:
		for i in xrange(4):
			um_new_feature[um][i+12] = (um_new_feature[um][i]*1.0)/um_new_count[um][i] if um_new_count[um][i] != 0 else 0
			um_new_feature[um][i+16] = (um_new_feature[um][i+4]*1.0)/um_new_count[um][i] if um_new_count[um][i] != 0 else 0
			um_new_feature[um][i+20] = (um_new_feature[um][i+8]*1.0)/um_new_count[um][i] if um_new_count[um][i] != 0 else 0
		um_new_feature[um][24] = len(user_new_stat[user][16]&merchant_new_stat[merchant][16])
		um_new_feature[um][25] = len(user_new_stat[user][17]&merchant_new_stat[merchant][17])

	print "add new feature to file"
	old_f = csv.reader(open(old_feature))
	if is_train:
		old_header = old_f.next()[:-1]
	else:
		old_header = old_f.next()
	fo = open(out_feature,'w')
	fo.write('user_id,merchant_id,')
	for it in old_header[2:]:
		fo.write(it+',')
	for i in xrange(n_new_um_feature):
		fo.write('numf'+str(i)+',')
	for i in xrange(n_new_user_feature):
		fo.write('nuf'+str(i)+',')
	if is_train:
		for i in xrange(n_new_merchant_feature):
			fo.write('nmf'+str(i)+',')
		fo.write('label\n')
	else:
		for i in xrange(n_new_merchant_feature):
			if i != n_new_merchant_feature-1:
				fo.write('nmf'+str(i)+',')
			else:
				fo.write('nmf'+str(i))
		fo.write('\n')

	for l in old_f:
		fo.write(l[0]+','+l[1]+',')
		if is_train:
			for feat in l[2:-1]:
				fo.write(feat+',')
		else:
			for feat in l[2:]:
				fo.write(feat+',')
		for feat in um_new_feature[(l[0],l[1])]:
			fo.write(str(feat)+',')
		for feat in user_new_feature[l[0]]:
			fo.write(str(feat)+',')
		for i in xrange(n_new_merchant_feature):
			if i != n_new_merchant_feature-1:
				fo.write(str(merchant_new_feature[l[1]][i])+',')
			else:
				fo.write(str(merchant_new_feature[l[1]][i]))
		if is_train:
			fo.write(','+um_label[(l[0],l[1])]+'\n')
		else:
			fo.write('\n')
	fo.close()

def ex_feature_v4(um_file,user_beh_log,out_feature):
	print "<user,merchant> pair: ",
	um_set = set()
	for e in util.parse(um_file):
		um_set.add((e['user_id'],e['merchant_id']))
	print len(um_set)

	user_subset = set([um[0] for um in um_set])
	print "user: %d"%(len(user_subset))

	merchant_subset = set([um[1] for um in um_set])
	print "merchant: %d"%(len(merchant_subset))
	double11 = date(2014,11,11)
	date_span1 = (date(2014,5,11),date(2014,7,11))
	date_span2 = (date(2014,7,12),date(2014,9,11))
	date_span3 = (date(2014,10,12),date(2014,11,10))
	# date_span4 = (date(2014,8,12),date(2014,9,11))
	# date_span5 = (date(2014,9,12),date(2014,10,11))
	# date_span6 = (date(2014,10,12),date(2014,11,10))

	# total_days = (date(2014,11,11)-date(2014,5,11)).days
	# span1_days = (date(2014,9,30)-date(2014,5,11)).days
	# span2_days = (date(2014,10,31)-date(2014,10,1)).days
	# span3_days = (date(2014,11,10)-date(2014,11,1)).days
	iv_um_feature = dict()
	iv_um_stat = dict()
	iv_um_count = dict()
	n_iv_um_f = 36

	iv_u_feature = dict()
	iv_u_stat = dict()
	iv_u_count = dict()
	n_iv_u_f = 38

	iv_m_feature = dict()
	iv_m_stat = dict()
	iv_m_count = dict()
	n_iv_m_f = 38

	print "extract time interval feature"
	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		item = e['item_id']
		cat = e['cat_id']
		brand = e['brand_id']
		dat = util.time_proc(e['time_stamp'])
		if dat > date(2014,11,11):
			dat = date(2014,11,11)
		act = int(e['action_type'])
		um = (user,merchant)
		if um in um_set:
			if um not in iv_um_feature:
				iv_um_feature[um] = [0 for i in xrange(n_iv_um_f)]
				iv_um_stat[um] = [set() for i in xrange(16)]
				iv_um_count[um] = [0 for i in xrange(12)]
			iv_um_stat[um][act].add(dat)
			if dat >= date_span1[0] and dat <= date_span1[1]:
				iv_um_count[um][act] += 1
				iv_um_stat[um][act+4].add(dat)
			if dat >= date_span2[0] and dat <= date_span2[1]:
				iv_um_count[um][act+4] += 1
				iv_um_stat[um][act+8].add(dat)
			if dat >= date_span3[0] and dat <= date_span3[1]:
				iv_um_count[um][act+8] += 1
				iv_um_stat[um][act+12].add(dat)

		if user in user_subset:
			if user not in iv_u_feature:
				iv_u_feature[user] = [0 for i in xrange(n_iv_u_f)]
				iv_u_stat[user] = [set() for i in xrange(28)]
				iv_u_count[user] = [0 for i in xrange(12)]
			iv_u_stat[user][act].add(dat)
			if dat >= date_span1[0] and dat <= date_span1[1]:
				iv_u_count[user][act] += 1
				iv_u_stat[user][act+4].add(dat)
				iv_u_stat[user][act+16].add(merchant)
			if dat >= date_span2[0] and dat <= date_span2[1]:
				iv_u_count[user][act+4] += 1
				iv_u_stat[user][act+8].add(dat)
				iv_u_stat[user][act+20].add(merchant)
			if dat >= date_span3[0] and dat <= date_span3[1]:
				iv_u_count[user][act+8] += 1
				iv_u_stat[user][act+12].add(dat)
				iv_u_stat[user][act+24].add(merchant)

		if merchant in merchant_subset:
			if merchant not in iv_m_feature:
				iv_m_feature[merchant] = [0 for i in xrange(n_iv_m_f)]
				iv_m_stat[merchant] = [set() for i in xrange(28)]
				iv_m_count[merchant] = [0 for i in xrange(12)]
			iv_m_stat[merchant][act].add(user)
			if dat >= date_span1[0] and dat <= date_span1[1]:
				iv_m_count[merchant][act] += 1
				iv_m_stat[merchant][act+4].add(dat)
				iv_m_stat[merchant][act+16].add(user)
			if dat >= date_span2[0] and dat <= date_span2[1]:
				iv_m_count[merchant][act+4] += 1
				iv_m_stat[merchant][act+8].add(dat)
				iv_m_stat[merchant][act+20].add(user)
			if dat >= date_span3[0] and dat <= date_span3[1]:
				iv_m_count[merchant][act+8] += 1
				iv_m_stat[merchant][act+12].add(dat)
				iv_m_stat[merchant][act+24].add(user)

	print "extract complete, begin stat"
	for user in user_subset:
		for i in xrange(4):
			iv_u_feature[user][i] = len(iv_u_stat[user][i+4])
			iv_u_feature[user][i+4] = len(iv_u_stat[user][i+8])
			iv_u_feature[user][i+8] = len(iv_u_stat[user][i+12])
			iv_u_feature[user][i+12] = (len(iv_u_stat[user][i+4])*1.0) / len(iv_u_stat[user][i]) if len(iv_u_stat[user][i]) != 0 else 0
			iv_u_feature[user][i+16] = (len(iv_u_stat[user][i+8])*1.0) / len(iv_u_stat[user][i]) if len(iv_u_stat[user][i]) != 0 else 0
			iv_u_feature[user][i+20] = (len(iv_u_stat[user][i+12])*1.0) / len(iv_u_stat[user][i]) if len(iv_u_stat[user][i]) != 0 else 0
			iv_u_feature[user][i+24] = len(iv_u_stat[user][i+16])
			iv_u_feature[user][i+28] = len(iv_u_stat[user][i+20])
			iv_u_feature[user][i+32] = len(iv_u_stat[user][i+24])
		iv_u_feature[user][36] = len(iv_u_stat[user][18]&iv_u_stat[user][22])
		iv_u_feature[user][37] = len(iv_u_stat[user][22]&iv_u_stat[user][26])

	for merchant in merchant_subset:
		for i in xrange(4):
			iv_m_feature[merchant][i] = len(iv_m_stat[merchant][i+4])
			iv_m_feature[merchant][i+4] = len(iv_m_stat[merchant][i+8])
			iv_m_feature[merchant][i+8] = len(iv_m_stat[merchant][i+12])
			iv_m_feature[merchant][i+12] = len(iv_m_stat[merchant][i+16])
			iv_m_feature[merchant][i+16] = len(iv_m_stat[merchant][i+20])
			iv_m_feature[merchant][i+20] = len(iv_m_stat[merchant][i+24])
			iv_m_feature[merchant][i+24] = (iv_m_feature[merchant][i+12]*1.0) / len(iv_m_stat[merchant][i]) if len(iv_m_stat[merchant][i]) != 0 else 0
			iv_m_feature[merchant][i+28] = (iv_m_feature[merchant][i+16]*1.0) / len(iv_m_stat[merchant][i]) if len(iv_m_stat[merchant][i]) != 0 else 0
			iv_m_feature[merchant][i+32] = (iv_m_feature[merchant][i+20]*1.0) / len(iv_m_stat[merchant][i]) if len(iv_m_stat[merchant][i]) != 0 else 0
		iv_m_feature[merchant][36] = len(iv_m_stat[merchant][18]&iv_m_stat[merchant][22])
		iv_m_feature[merchant][37] = len(iv_m_stat[merchant][22]&iv_m_stat[merchant][26])

	for um in um_set:
		for i in xrange(4):
			iv_um_feature[um][i] = iv_um_count[um][i]
			iv_um_feature[um][i+4] = iv_um_count[um][i+4]
			iv_um_feature[um][i+8] = iv_um_count[um][i+8]
			iv_um_feature[um][i+12] = len(iv_um_stat[um][i+4])
			iv_um_feature[um][i+16] = len(iv_um_stat[um][i+8])
			iv_um_feature[um][i+20] = len(iv_um_stat[um][i+12])
			iv_um_feature[um][i+24] = (len(iv_um_stat[um][i+4])*1.0) / len(iv_um_stat[um][i]) if len(iv_um_stat[um][i]) != 0 else 0
			iv_um_feature[um][i+28] = (len(iv_um_stat[um][i+8])*1.0) / len(iv_um_stat[um][i]) if len(iv_um_stat[um][i]) != 0 else 0
			iv_um_feature[um][i+32] = (len(iv_um_stat[um][i+12])*1.0) / len(iv_um_stat[um][i]) if len(iv_um_stat[um][i]) != 0 else 0
	
	print "write to file"
	fo = open(out_feature,'w')
	fo.write('user_id,merchant_id,')
	for i in xrange(n_iv_um_f):
		fo.write('ivumf'+str(i)+',')
	for i in xrange(n_iv_u_f):
		fo.write('ivuf'+str(i)+',')
	for i in xrange(n_iv_m_f):
		if i != n_iv_m_f-1:
			fo.write('ivmf'+str(i)+',')
		else:
			fo.write('ivmf'+str(i)+'\n')
	for um in iv_um_feature:
		fo.write(um[0]+','+um[1]+',')
		for feat in iv_um_feature[um]:
			fo.write(str(feat)+',')
		for feat in iv_u_feature[um[0]]:
			fo.write(str(feat)+',')
		for i in xrange(n_iv_m_f):
			if i != n_iv_m_f-1:
				fo.write(str(iv_m_feature[um[1]][i])+',')
			else:
				fo.write(str(iv_m_feature[um[1]][i])+'\n')
	fo.close()


def out_similar_merchant(user_beh_log,similar_file):
	# Compute the top 5 similar partners for each merchant.
	# The similarity is simply defined by normalized number of overlapping customers
	print "Compute the top 5 similar partners for each merchant"
	merchant_customer = dict()
	for e in util.parse(user_beh_log):
		user = e['user_id']
		merchant = e['seller_id']
		act = int(e['action_type'])
		if merchant not in merchant_customer:
			merchant_customer[merchant] = set()
		if act == 2:
			merchant_customer[merchant].add(user)

	num_merchant = len(merchant_customer)
	merchant_list = np.array(merchant_customer.keys())
	print "merchant num: %d"%(len(merchant_list))
	m2m_sim = np.zeros((num_merchant,num_merchant))
	for i in xrange(num_merchant):
		for j in xrange(num_merchant):
			if i != j:
				m2m_sim[i][j] = len(merchant_customer[merchant_list[i]]&merchant_customer[merchant_list[j]])

	# m2m_sim = preprocessing.StandardScaler().fit_transform(m2m_sim)
	m2m_sim = preprocessing.normalize(m2m_sim,norm='l2')
	merchant_sim_m = dict()
	for i in xrange(num_merchant):
		top_k = merchant_list[np.argsort(m2m_sim[i,:])[-5:]]
		merchant_sim_m[merchant_list[i]] = top_k

	# print merchant_sim_m

	fo = open(similar_file, "w")
	fo.write('merchant_id,sim_m1,sim_m2,sim_m3,sim_m4,sim_m5\n')
	for i in xrange(num_merchant):
		fo.write(merchant_list[i]+',')
		for k in xrange(5):
			if k != 4:
				fo.write(merchant_sim_m[merchant_list[i]][k]+',')
			else:
				fo.write(merchant_sim_m[merchant_list[i]][k]+'\n')

	fo.close()
	return m2m_sim

def feature_combine(my_feature,other_feature,feature_list,out_feature):
	my_fi = csv.reader(open(my_feature))
	my_header = my_fi.next()
	my_is_feature = [i for i in xrange(len(my_header))]
	my_is_feature.remove(0)
	my_is_feature.remove(1)
	if 'label' in my_header:
		my_is_feature.remove(my_header.index('label'))
		label_index = my_header.index('label')
		flag = True
	else:
		flag = False

	ot_fi = csv.reader(open(other_feature))
	ot_header = ot_fi.next()
	ot_is_feature = [ot_header.index(f) for f in feature_list]
	# ot_is_feature.remove(0)
	# ot_is_feature.remove(1)
	# if 'label' in ot_header:
	# 	ot_is_feature.remove(ot_header.index('label'))

	myf = dict()
	label = dict()
	for ml in my_fi:
		um = (ml[0],ml[1])
		myf[um] = [ml[k] for k in my_is_feature]
		if flag:
			label[um] = ml[label_index]
	otf = dict()
	for ol in ot_fi:
		um = (ol[0],ol[1])
		otf[um] = [ol[k] for k in ot_is_feature]

	fo = open(out_feature,'w')
	fo.write('user_id,merchant_id,')
	for i in my_is_feature:
		fo.write(my_header[i]+',')
	for i in ot_is_feature:
		if i != ot_is_feature[-1]:
			fo.write(ot_header[i]+',')
		else:
			fo.write(ot_header[i])
	if flag:
		fo.write(',label\n')
	else:
		fo.write('\n')
	for um in myf:
		fo.write(um[0]+','+um[1]+',')
		for feat in myf[um]:
			fo.write(feat+',')
		if flag:
			for feat in otf[um]:
				fo.write(feat+',')
			fo.write(label[um]+'\n')
		else:
			for i in xrange(len(otf[um])):
				if i != len(otf[um]) - 1:
					fo.write(otf[um][i]+',')
				else:
					fo.write(otf[um][i]+'\n')
	fo.close()