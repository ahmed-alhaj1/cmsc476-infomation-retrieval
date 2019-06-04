#author : ahmed alhaj
#email : alhaj1@umbc.edu
#class: 476 Inoformation Retrival
#proj : getting the frequency of tokens in several files




from __future__ import division, unicode_literals
import nltk
import os
import codecs
import html2text
from collections import Counter
import re
import time
import numpy as np
from matplotlib import pyplot as plt





def set_freq(text, path):

    freq_file = open(path+"freq.txt", "w", errors = 'ignore')

    process_text = "".join(text)
    #words = re.findall(r'\w', process_text)
    #for i in text:

    unwanted_char =  ['0.00','0.01', '0.02','\n','|','\\','`','*','_','{','}','[',']','(',')','>','#','+','-','.','!','$']
    for ch in unwanted_char:
        if ch == '|':
            process_text = process_text.replace(ch, "", 98939)
        elif ch in process_text:
            #print("doing it")
            process_text = process_text.replace(ch, "",70000 )
    freq_dist = Counter(process_text.split())



    #freq_dist = nltk.FreqDist(text)

    #for k in freq_dist.items():
    s = [(k, freq_dist[k]) for k in sorted(freq_dist, key=freq_dist.get, reverse=True)]
    for k, v in s:
        #k, v
        #print(k)
        freq_file.write("%s \n"% str(k))
        freq_file.write("\t")
        freq_file.write("%s \n"% v)

    freq_file.close()
    print(freq_dist.most_common(20))

    token_file = open(path+"token.txt", "w", errors = 'ignore')
    for key in sorted(freq_dist.keys()):
        token_file.write("%s \t"% str(key))
        token_file.write("\t")
        token_file.write("%s \n"% freq_dist[key])

    return



def main():
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
    path1 =  r'''C:\Users\alhaj1\Documents\cmsc476\out_files\outfile'''

    files  = os.listdir(path)
    #try:
    #os.mkdir(os.path.dirname(path1))
    #except OSError as exc:
    #print("somthing wrong happend ")
    end1 = 0
    all_text = list()
    i = 0;
    freq_list = []
    for file in files:
        if(file.endswith('.html')):
            fname = os.path.join(path,file)
            in_file = open(fname, 'r', errors ='ignore' );


            tkz_text = nltk.sent_tokenize(h.handle(in_file.read().lower())) #
            #print(type(tkz_text))
            all_text += tkz_text

            in_file.close()
            outfile = open(path1+str(i)+'.txt', 'w');
            outfile.write(str(tkz_text))
            outfile.close()
            time_list.append(start - time.time())
            end1 += time.time()
            i= i+1

        #####################################
    set_freq(all_text, path)
    ## here I am just plot
    """
    end2 = time.time()
#plotting the time
    ratio = (end2 - end1) / 504

    #time_list.append(start-time.time())
    time_list = np.asarray(time_list)
    time_list = time_list +ratio

    #time_list = np.full( (503) ((start-time.time())/503))
    #print(time_list)
    num_files = np.arange(503)

    plt.plot(num_files, time_list )
    plt.show()
    """
    print(start - time.time())
main()
