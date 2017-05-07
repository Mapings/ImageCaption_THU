# -*- coding: utf-8 -*-
from string import digis
import re

#Read original data
file_train = open('train.txt')
data = file_train.read( ).replace("\n","").replace("\r","").replace(" ","")
temp = list(re.sub("[^a-zA-Z]+", "", data))

print("length of original data:", len(data))
for i in range(len(temp)):
    data = data.replace(str(temp[i]), "")
print("length of processed data:", len(data))

data = ''.join(set(data))

print("length of final data:", len(data))

file_train.close( )

f1 = open('train_new.txt','w')
f1.write(data)
f1.close( )