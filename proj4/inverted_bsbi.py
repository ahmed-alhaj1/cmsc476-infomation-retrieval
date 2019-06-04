
#######################################################
# author : ahmed alhaj
# email : alhaj1@umbc.edu
# phase2: caculate the frequency of token with respect to fileself
# class : cmsc476Block
#######################################################






import re
import glob
import collections
import math
import pickle
import html2text
from structures import WordParser, FilesPiper, HtmlFile, InvdIdx as InvdIndx
import time
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt



"""
this is the main method where everything is running

"""


def block_sort_based_indexing(path=None, common_word=None):
#def term_weighter(path=None, path1=None, common_words_path=None):
    #stop_word = []
    #path = r'''C:\Users\alhaj1\Documents\cmsc476\files'''

    if (path == None):

        path = r'''C:\Users\alhaj1\Documents\cmsc476\files'''
        path1 =  r'''C:\Users\alhaj1\Documents\cmsc476\proj2\proj2\out_files3\doc'''
        common_words_path = r'''C:\Users\alhaj1\Documents\cmsc476\proj2\stop_list.txt'''

    inverted_list = []
    retrieval_list = []
    htfl_block = {}

    parser = WordParser(common_words_path)
    files_piper = FilesPiper(path)
    invd_indx = InvdIndx()


    i = 0
    j = 0
    time_list = []
    consumed_time = 0
    for htfl in files_piper.get_files():
        if (i % 100 == 0):
            if len(inverted_list) > 0 :
                invd_indx.calc_tf_idf()


            inverted_list.append(invd_indx)
            inverted_list[j].calc_tf_idf()
            invd_indx = InvdIndx()
            retrieval_list.append(htfl_block)
            htfl_block = {}
            j+=1


        htfl.tokenize(parser, invd_indx)
        htfl_block[htfl.id] = htfl.to_string()
        i +=1
        time_list.append(time.time())

    doc_retrieval = retrieval_list[0]
    for htfl_block in retrieval_list:
        doc_retrieval = {**doc_retrieval, ** htfl_block}
    for inverted_index in inverted_list[1:]:
        inverted_list[0].contactenate_indexes(inverted_index)
    inv_indx =inverted_list[0]
    #with open("output3.txt", "w") as f:
    #    f.write(str(inverted_list[0]))
    inverted_list[0].save_inverted_index(path=None)
    #return (inverted_list[0], doc_retrieval)
    retrieval_collections = "retrieval_documents.p"
    with open(retrieval_collections, 'wb') as f:
        pickle.dump(doc_retrieval, f, pickle.HIGHEST_PROTOCOL)



def main():

    path1 = None;  path2 = None ; path3 = None
    input_list = []
    for i in range(3):
        if i == 0:
            input_list.append(input("please enter the in_file path"))
        elif i == 1:
            input_list.append(input("please enter the out_file path"))
        elif i == 2:
            input_list.append(input("please enter the common_word path"))

    if(len(input_list[0]) > 1):
        path1 , path2 , path3 = input_list[0], input_list[1], input_list[2]
    start = time.time()
    block_sort_based_indexing(path1, path2)
    exit()
    #map_reduce(map_parser,reducer)
    num_files = np.arange(0,503)
    end = time.time()

    term_to_search = input("please enter term to search ")
    docs_list = invert_index.look_term_index(term_to_search)
    if docs_list == -1:
        print("something wroing happen")
        exit()
    print(docs_list, len(docs_list))
    #time_list = []
    start = time.time()
    for i in docs_list:
        if i in doc_retrieval.keys():

            for x,y in enumerate(doc_retrieval[i].split()):
                if y == term_to_search :
                    print(x, "------>" , y)


    #for (x,y) in invert_index.
    end = time.time()
    print(end- start)
    print("time took to search for the element =", end-start)
    filelist = np.arange(0,503)
    #timelist = np.repeat(((end - start)/503), 503)

    #t = 0
    #final_time_list = []
    #for i in time_list:
    #    t+=i
    #    final_time_list.append(t)
    #plt.plot(num_files, time_list)
    #print("time list ", final_time_list)
    #print("filelist ", filelist)
    #plt.show()




main()
