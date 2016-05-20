import os
import sys
import math

def sigmoid(x):
    r=x
    if x > 19:
        r=19
    if x<-19:
        r=-19
    return 1.0 / (1 + math.exp(r))

def be_positive(x):
    return sigmoid(-x)

def be_negative(x):
    return 1.0-be_positive(x)

def dot_product(x,y):
    assert(len(x)==len(y))
    result=0.0
    for i in range(len(x)):
        result+=x[i]*y[i]
    return result


dimensionNumber=15

if __name__=="__main__":
    train_data_file_name="train_data.txt"
    train_label_file_name="train_label.txt"
    test_data_file_name="test.txt"
    weight_file_name="weights.txt"
    
    train_label=[]
    fp=open(train_label_file_name,'r')
    for line in fp:
        train_label.append(int(line.strip()))
    fp.close()
    
    weights=[]
    
    fp=open(weight_file_name,'r')
    for line in fp:
        weights.append(float(line.strip()))
    fp.close()
    
    train_data=[]
    
    fp=open(train_data_file_name,'r')
    for line in fp:
        if len(line.strip())>0:
            single_data=map(float,line.strip().split(' '))
            assert(len(single_data)==len(weights)/2)
            train_data.append(single_data)
    fp.close()
    
    test_data=[]
    
    fp=open(test_data_file_name,'r')
    for line in fp:
        if len(line.strip())>0:
            single_data=map(float,line.strip().split(' '))
            assert(len(single_data)==len(weights)/2)
            test_data.append(single_data)
    fp.close()
    assert(len(train_data)==len(train_label))
    
    L=len(train_data)
    for i in range(len(test_data)):
        test_instance=test_data[i]
        result=0.0
        for j in range(len(train_data)):
            train_instance=train_data[j]
            sample_instance=test_instance+train_instance
            if train_label[j]==1:
                result+=be_negative(dot_product(sample_instance,weights))
            else:
                result+=be_positive(dot_product(sample_instance,weights))
        print result
            
    
    