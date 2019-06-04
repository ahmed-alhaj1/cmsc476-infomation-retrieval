import math
import collections
import re

#from query import Tree

class TokenVector:
    def __init__(self):
        self.v = {}
        self.norm = -1
    def setitem(self, k, v):
        self.v[k] = v
        self.norm = -1
    def setNorm(self, value) :
        self.norm = value
    def normalized(self):
        if(self.norm ==-1):
            self.norm = math.sqrt(sum([v * v for (k,v) in self.v.items()]))
        return self.norm
    def cosSimilarity(self, other):
        v1 = self.v
        v2 = other.v
        vect1_dims = set(v1.keys())
        vect2_dims = set(v2.keys())
        shared_dims = vect1_dims.intersection(vect2_dims)
        #print("shared dimension -> ",shared_dims)
        num = sum([v1[dim] * v2[dim] for dim in shared_dims])
        return num/(self.normalized() *other.normalized())
    def cosCorrespondancesDims(self, other):
        print ("v ->", self.v, "\n", "other ->" ,other.v )
        num = sum([self.v[dim] * other.v[dim] for dim in self.v.keys()])
        return num // ( self.normalized() * other.normalized() )

class VectorModel:
    def __init__(self, method=None):
        self.technique = method

    def search(self, input, invd_index, word_tokenizer):
        if self.technique == None:
            dc_norms = invd_index.dc_nrm_tf_idf

        gen = word_tokenizer.Parse(input)
        parsed_query = [tk for tk in gen]

        query_vect = TokenVector()
        counter = collections.Counter(parsed_query)
        print(counter)
        for (tk, amt) in counter.items():
            if tk in invd_index.invd_indx:
                idf = math.log10(len(invd_index.invd_indx)/len(invd_index.invd_indx[tk]))
                print(idf)
                query_vect.v[tk]= (1+math.log10(amt))* idf
        dc_vects = collections.defaultdict(TokenVector)
        for tm in invd_index.invd_indx:
            postings = invd_index.invd_indx[tm]
            idf = math.log10(len(invd_index.invd_indx)/ len(postings))
            for (dc_id, tf) in postings.items():
                dc_vects[dc_id].v[tm] = (1+ math.log10(tf)) * idf


        for (dc_id, dc_vect) in dc_vects.items():
            #print("dc norm =", dc_norms[dc_id])
            dc_vect.setNorm(math.sqrt(dc_norms[dc_id]))
        similarities = {dc_id: doc_vect.cosSimilarity(query_vect) for dc_id,doc_vect in dc_vects.items() }
        sorted_doc_ids = sorted(similarities, key=lambda k:similarities[k], reverse=True)
        return sorted_doc_ids
