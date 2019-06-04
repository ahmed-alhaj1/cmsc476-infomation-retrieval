import re
import glob
import collections
import math
import pickle

import numpy as np

class InvdIdx:
    def __init__(self, functions= None):

        # this member supposed to be useful for trying other formal
        #self.funcs = functions
        # it nested dicts {token: [{fl_id: tk_freq },{fl_id:tk_freq},{}]}
        self.invd_indx = collections.defaultdict(lambda: collections.defaultdict(int))

        # it is not used but I mean to have it document with high frequency for fuutre
        # functions to do some calcution or filtering if possible
        self.dc_nrm_tf_idf = collections.defaultdict(int)
        self.dc_high_frq = collections.defaultdict(int)
        self.tm_nrm_tf_idf = collections.defaultdict(float)



    """
    normal str_member for print and and writing the inverted index
    """
    def __str__(self):
        res= ""

        for(k, v) in self.invd_indx.items():
            res += k +"\t ->  " +str( {x: y for (x,y) in v.items()}) +"\t"+str(self.dc_nrm_tf_idf[k]) + "\n"
        return res

    def to_string(self):
        for (k,v) in self.dc_nrm_tf_idf.items():
            print(k,v)

    def contactenate_indexes(self, new_index):
        for wd in new_index.invd_indx.keys():
            self.invd_indx[wd].update(new_index.invd_indx[wd])

        self.dc_nrm_tf_idf.update(new_index.dc_nrm_tf_idf)
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

    def calc_term_tf_idf(self):
        for tm, tm_qnty in self.invd_indx.items():

            #if len(tm_qnty.keys()) > 1:
            tm_freq = 0
            for (x,y) in tm_qnty.items():
                tm_freq = y
            nrm_idf = 0.4 + 0.6* float( tm_freq/len(self.invd_indx))
            self.tm_nrm_tf_idf[tm] = nrm_idf

    """
    def remove_token(self):
        for (k, v) in self.invd_idx.items():
            if len(v) <= 1:
                invd_index.
    """
