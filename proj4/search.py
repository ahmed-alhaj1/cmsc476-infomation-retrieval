import argparse
import pickle
import re
import time


from structures import WordParser, FilesPiper, HtmlFile, InvdIdx as invd_index
from vector_space import TokenVector, VectorModel


def pad(str):
    return re.sub('^', ' '*4, str, flags=re.MULTILINE)
def Search_vector_space_model():
    path_retrieval = "retrieval_documents.p"
    sample_file = "search_result.txt"
    #path = r'''C:\Users\alhaj1\Documents\cmsc476\files'''
    path1 =  r'''C:\Users\alhaj1\Documents\cmsc476\proj2\proj2\out_files3\doc'''
    common_words_path = r'''C:\Users\alhaj1\Documents\cmsc476\proj2\stop_list.txt'''

    #query_input = input("please enter your query, quit to exit")

    parser = WordParser(common_words_path)
    #files_piper = FilesPiper(path)
    #invd_indx = ()


    model = VectorModel(None)
    with open(path_retrieval, 'rb')  as f:
        retrieved_dcs = pickle.load(f)

        inv_index = invd_index()
        inv_index.read_inverted_index(None)

        while(1):

            query_input = input("please enter your query, quit to exit")
            start = time.time()
            if(query_input == "quit"):
                return
            dc_ids = inv_index.find(query_input, model, parser)
            with open(sample_file, "w") as f:
                for i , dc_id in enumerate(dc_ids[:10]):
                    print('Document '+ str(i+1) + ':#'+ str(dc_id))
                    f.write(str(dc_id) + str("####################################################\n") + pad(retrieved_dcs[dc_id]) + str('\n'))
                #print(retrieved_dcs[dc_id])
                print("time take to retrieval document :", (time.time() - start) )

if __name__ ==  "__main__":
    print("searching")
    start = time.time()
    Search_vector_space_model()
    end = time.time()
    print("time elaspsed = ", (end -start))
