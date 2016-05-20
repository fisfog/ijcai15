import os

import sys


if __name__=="__main__":
    train_data_file_name="train_data.txt"
    train_label_file_name="train_label.txt"
    gen_data_file_name="gen.txt"
    
    train_label=[]
    fp=open(train_label_file_name,'r')
    for line in fp:
        train_label.append(int(line.strip()))
    fp.close()
    
    
    train_data=[]
    
    fp=open(train_data_file_name,'r')
    for line in fp:
        if len(line.strip())>0:
            single_data=map(float,line.strip().split(' '))
            train_data.append(single_data)
    fp.close()
    
    assert(len(train_data)==len(train_label))
    
    L=len(train_data)
    fp=open(gen_data_file_name,'w')
    for i in range(0,L-1):
        for j in range(i+1,L):
            gen_data=train_data[i]+train_data[j]
            label=0 if (train_label[i]==train_label[j]) else 1
            fp.write(' '.join(map(str,gen_data))+' '+str(label)+'\n')
    
    fp.close()