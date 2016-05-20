#-*- coding:utf-8 -*-

"""
	littlekid
	muyunlei@gmail.com
"""
# import gzip
import numpy as np
from datetime import *


def parse(csvfile):
	"""
	解析数据
	"""
	f = open(csvfile, 'r')
	head = f.readline().strip('\r\n').split(',')
	n = len(head)
	entry = {}
	for l in f:
		l = l.strip('\r\n').split(',')
		for i in xrange(n):
			entry[head[i]] = l[i]
		yield entry

def time_proc(timestr):
	return date(2014,int(timestr[:2]),int(timestr[2:]))

def output_result(result,outfile):
	f = open(outfile,'w')
	f.write('user_id,merchant_id,prob\n')
	for um in result:
		f.write(um[0]+','+um[1]+','+str(result[um])+'\n')
	f.close()
