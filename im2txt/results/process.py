#coding = utf-8

# import os



f_submit = open('submit_attention_22.txt','r')
captions = list()
for line in f_submit.readlines():
	line = line.replace("  ","")
	captions.append(line)
f_submit.close()
open('submit_attention_22_new.txt', 'w').write(''.join(captions))


