from __future__ import division, unicode_literals
import nltk
import os
import codecs
import html2text
from collections import Counter
from nltk.corpus import stopwords
import time
import numpy as np
from matplotlib import pyplot as plt







set(stopwords.words('english'))


start = time.time()
time_list = []
print("Hello World ! \n")

h = html2text.HTML2Text()
h.ignore_links = True

input_dir = str()
output_dir = str()

#input_dir = input("please enter the input directroy \n" )
path1= input_dir

path = r'''C:\Users\alhaj1\Documents\cmsc476\files'''
path1 =  r'''C:\Users\alhaj1\Documents\cmsc476\out_files2\outfile2'''

end1 = 0
files  = os.listdir(path)
#try:
#os.mkdir(os.path.dirname(path1))
#except OSError as exc:
    #print("somthing wrong happend ")
i = 0;
tot_text = str()
freq_dist = dict()
for file in files:
    if(file.endswith('.html')):
        fname = os.path.join(path,file)
        in_file = open(fname, 'r', errors ='ignore' );
        file_to_clean = h.handle(in_file.read())
        unwanted_char =  ['0.00','0.01', '0.02','\n','|','\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$']
        for ch in unwanted_char:
            if ch == '|':
                file_to_clean = file_to_clean.replace(ch, "", 200)
            elif ch in file_to_clean:
                file_to_clean = file_to_clean.replace(ch,"", 200)

        tot_text += file_to_clean
        in_file.close()
        time_list.append(start-time.time())
        i+=1



print(i)

end1 = start - time.time()

arr = []

tokens = nltk.word_tokenize(tot_text)
freq_dist = nltk.FreqDist(tokens)
outfile = open(path1+str(i)+'.txt', 'w', errors = 'ignore');
s = [(k, freq_dist[k]) for k in sorted(freq_dist, key=freq_dist.get, reverse=True)]
for k, v in s:
    #k, v
    #print(k)
    outfile.write("%s \t"% k)
    outfile.write("%s \n"% v)


outfile.close()

end = time.time()
#time_list.append(start -end)

time_list = np.asarray(time_list)
num_files = np.arange(503)

ratio = (end1 - end) / 503

time_list += ratio


plt.plot(num_files, time_list)
plt.show()
