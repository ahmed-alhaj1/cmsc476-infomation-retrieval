import re
import glob
import collections
import math
import pickle
import ctypes

import numpy as np

class InvdIdx:
    def __init__(self, functions= None):

        # this member supposed to be useful for trying other formal
        #self.funcs = functions
        # it nested dicts {token: [{fl_id: tk_freq },{fl_id:tk_freq},{}]}
        self.invd_indx = collections.defaultdict(lambda: collections.defaultdict(int))

        # it is not used but I mean to have it document with high frequency for fuutre
        # functions to do some calcution or filtering if possible
        self.dc_nrm_tf_idf =  collections.defaultdict(int)
        self.dc_high_frq = collections.defaultdict(int)
        self.tm_nrm_tf_idf = collections.defaultdict(float)
        self.doc_most_frequent = collections.defaultdict(int)
        self.doc_norm_norm_freq = collections.defaultdict(int)


    """
    normal str_member for print and and writing the inverted index
    """

    def __str__(self):
        #print(len(self.dc_nrm_tf_idf))
        #ff = "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ /n /n /n /n"
        #print(ff*13)
        #for s, n in self.dc_nrm_tf_idf.items():
        #    print(s,n)

        #ff = "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ /n /n /n /n"
        #print(ff*13)
        res= ""


        for k in self.invd_indx.keys():
            #if (k in list(self.dc_nrm_tf_idf.keys.
            res += k +"\t ->  " + str({x: y for (x,y) in self.invd_indx[k].items()}) +"\t" + " --> " +str( {x: self.dc_nrm_tf_idf[x] for (x,y) in self.invd_indx[k].items()}) +"\n"
            #try:
            #     list(self.dc_nrm_tf_idf.keys()).index(k)
            #except:
            #     pass   #str(list(self.dc_nrm_tf_idf.keys()).index(k)) #+ "\t" + str(z  for m in self.invd_indx[k].keys() for z in self.dc_nrm_tf_idf.get(m))+ "\n"
        return res

    def to_string(self):
        for (k,v) in self.dc_nrm_tf_idf.items():
            print(k,v)

    def contactenate_indexes(self, new_index):
        for wd in new_index.invd_indx.keys():
            self.invd_indx[wd].update(new_index.invd_indx[wd])

        self.dc_nrm_tf_idf.update(new_index.dc_nrm_tf_idf)
        #print("print the weight ")
        x = "########################################### \n"
        #print(x * 12)
        #print(self.dc_nrm_tf_idf.keys())
        #print(self.dc_nrm_tf_idf.values())
        #self.doc_norms_tf_idf_norm.update(inv_index.doc_norms_tf_idf_norm)

    """
    inscribe_token:
    input: token, document_id
    postcondition :registes the token in the inverted index
    """
    def inscribe_token(self, token, dc_id):
        self.invd_indx[token][dc_id] += 1



    """
    calc_tf_idf:
        using tf(t, d, D) * idf(t,d)
        A high weight in tf–idf is reached by a high term frequency and a
        low document frequency of the term
    """

    """
    dump_term_weight: dumps out term weight for each document
    """
    def dump_term_weight(self):
         res = str()
         for x, y in self.tm_nrm_tf_idf.items():
             new_res = str(x) + "\t ->"+ str(y) + "\n"
             #print("#########" ,res )
             res = res + new_res
         return res


    def calc_tf_idf(self):
        for tm, tm_qnty in self.invd_indx.items():
            idf = np.log10(len(self.invd_indx) / len(tm_qnty))
            for dc_id, tf  in tm_qnty.items():

                tfidf = tf* idf
                self.dc_nrm_tf_idf[dc_id]+=(tfidf**2)
                #print(self.dc_nrm_tf_idf[dc_id])

    def dump_weight(self):
        for dc_id, wgh in self.doc_norm_norm_freq.items():
            print(dc_id , wgh)


    def calc_term_tf_idf(self):
        for tm, tm_qnty in self.invd_indx.items():
            idf = math.log10(len(self.invd_indx)/len(tm_qnty))
            tm_freq = 0
            for dc,prev_tf in tm_qnty.items():
                tf = 1+ math.log(prev_tf)
                tf_idf = tf * idf
                self.tm_nrm_tf_idf[dc] += tf_idf
                #print(self.tm_nrm_tf_idf[dc])
    def look_term_index(self, term):
        if term in self.invd_indx.keys():
            return list(self.invd_indx[term].keys())

        return -1




    def synthenize_term_frequency(self):
        term_idx = collections.defaultdict(lambda:collections.defaultdict(int))
        for tm , postings in self.invd_indx.items():
            for dc_id, amt in postings.items():
                term_idx [dc_id][tm] += amt
        for dc_id , wds in term_idx.items():
            self.doc_most_frequent[dc_id] = max(wds.values())
            for word , raw_tf in wds.items():
                self.doc_norm_norm_freq[dc_id] += (raw_tf/ self.doc_most_frequent[dc_id])
