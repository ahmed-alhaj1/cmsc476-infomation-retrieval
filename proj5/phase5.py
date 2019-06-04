import pickle
from structures import WordParser, FilesPiper, HtmlFile, InvdIdx # as invd_index
from vector_space import TokenVector
import copy
import numpy as np
import collections
import math
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
#from scipy import cluster.hierarchy as sch
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import ward, dendrogram
def build_similarity_matrix(invd_index):
    H , W = 503, 503
    dc_norms = invd_index.dc_nrm_tf_idf
    similarity_matrix = np.zeros((H,W))
    doc_vectors  = collections.defaultdict(TokenVector)
    for tm in invd_index.invd_indx:
        postings = invd_index.invd_indx[tm]
        idf = math.log10((len(invd_index.invd_indx)/len(postings)))
        for(dc_id, tf) in postings.items():
            doc_vectors[dc_id].v[tm] = (1+ math.log10(tf)) * idf

    #doc_vectors2 = copy.deepcopy(doc_vetors1)
    deg = 0
    similar_docs = []
    for dc_id1 , dc_vect1 in doc_vectors.items():

        for dc_id2, dc_vect2 in doc_vectors.items():

            similarity = dc_vect1.cosSimilarity(dc_vect2)
            if (similarity > 0.4):
                deg = 1
            else :
                deg = 0
            similarity_matrix[int(dc_id1), int(dc_id2)] = deg
            if deg ==1 and dc_id1 != dc_id2 :
                similar_docs.append((int(dc_id1), int(dc_id2)))

    return similar_docs , similarity_matrix

def do_clustering(similarity_matrix):
    model = AgglomerativeClustering(affinity= 'precomputed', n_clusters=20, linkage='average' ).fit(similarity_matrix)

    #clustering = model.fit(similarity_matrix)
    titles1 = [i for i in range(1,503)]
    m = "#" *20
    print(m, '\n', m)
    print(model.labels_)
    print(m,'\n', m)
    print(model.n_leaves_)
    print(model.n_clusters)
    Z = hierarchy.linkage(similarity_matrix, 'single')
    fig ,axes= plt.subplots(1,figsize=(15,20))
    plt.title("custermerDenograms")
    dend = hierarchy.dendrogram( Z, ax=axes, above_threshold_color='y', orientation="top" )
    plt.tick_params(axis='', which='both', bottom='off', labelbottom='off')
    plt.tight_layout()
    plt.savefig('documents_cluster.png', dpi=503)
    plt.show()
    #plt.axes([])


def trigger_clustering():

    path_retrieval = "retrieval_documents.p"
    sample_file = "search_result.txt"
    path1 =  r'''C:\Users\alhaj1\Documents\cmsc476\proj2\proj2\out_files3\doc'''
    common_words_path = r'''C:\Users\alhaj1\Documents\cmsc476\proj2\stop_list.txt'''
    invd_index = InvdIdx()
    print("similarity matrix ")
    with open(path_retrieval, 'rb') as f:
        retrieved_dcs = pickle.load(f)
        invd_index.read_inverted_index(None)

    similar_dcs, similarity_matrix = build_similarity_matrix(invd_index)
    with open('similarity.txt', 'w') as f:
        for i in similar_dcs:
            x, y = i
            f.write(str(x) + " : " + str(y) + '\n')

    #for i in range(503):
    #    print(similarity_matrix[i, :] )
    #print("\n", similarity_matrix.shape)

    do_clustering(similarity_matrix)
    #print(similar_dcs)
    #print(len(similar_dcs))

if __name__ == "__main__":
    print ("trigger clustering is running ")
    trigger_clustering()
