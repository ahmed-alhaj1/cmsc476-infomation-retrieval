import os
from extractor_models import De_Conv_Autoencoder, plot_sample_imgs, plot_encoded_imgs
import numpy as np
from skimage import io
from skimage import img_as_float
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.externals import joblib
from load_data import load_dataset, FeaturesIndex
from skimage import img_as_float
import collections
from sklearn.cluster import KMeans
import cv2
import h5py
from matplotlib import pyplot as plt

#from build_index import FeaturesIndex





def extract_feat_query(encoder_model, img):
    H,W = img.shape
    img = img.reshape(1,H,W,1)
    #img =np.array([img])
    #print(img.shape)
    #img = np.array([img])
    #H,W = img.shape
    feats = encoder_model.predict(img)
    for feat in feats:
        feat_vect = feat / np.linalg.norm(feat)
        print("-------->",len(feat_vect.flatten()))
        return feat_vect


def cluster_query_img(feat_vect):
    n_clusters = 100
    file_name = os.path.join(os.getcwd(), "kmeans_data.pkl")
    #kmeans_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    kmeans_model = joblib.load( file_name)
    print(feat_vect.shape)
    H, W, ny= feat_vect.shape

    feat_vect = feat_vect.reshape(H,W)
    predicted_cluster = kmeans_model.fit_predict(feat_vect)
    print("predicted_cluster ------->", predicted_cluster)
    #exit()
    return predicted_cluster



def plot_sample_imgs(x_test, n):
    fig = plt.figure(figsize=(10,10), dpi=100)
    for i, k in enumerate(n):
        i = i+1
        fig.add_subplot(2, len(n)/2,i)
        plt.imshow(x_test[i].reshape(124,124))
        plt.gray()
        #ax.set_axis_off()
        plt.savefig("result.jpg")
    plt.show()
    plt.close(2)



def test_cbir():
    x_train, x_test, labels = load_dataset(os.getcwd(), False)
    encoder_model = De_Conv_Autoencoder()
    encoder_model.build_auto_encoder()
    encoder_model.compile()
    encoder_model.load()

    query_img = img_as_float(cv2.cvtColor(cv2.resize(cv2.imread( "test_img.jpg"), (124,124)), cv2.COLOR_BGR2GRAY))
    #print(type(query_img)
    print(query_img.shape)
    img_feat_vect  = extract_feat_query(encoder_model, query_img)
    predicted_cluster = cluster_query_img(img_feat_vect)

    plt.subplot(2,1,1)
    plt.imshow(query_img.reshape(124,124))
    plt.gray()
    #plt.set_axis_off()
    plt.show()
    plt.close()
    plt.savefig("sample.jpg")

    # initlize feature index
    index = FeaturesIndex()
    index.load()
    for k in index.feat_index:
        print(k," ---> ", index.feat_index[k])
    print(predicted_cluster)
    imgs_in_cluster = index.feat_index.get(predicted_cluster[0])
    print(list(set(imgs_in_cluster)))
    print(len(list(set(imgs_in_cluster))))
    plot_sample_imgs(x_train, list(set(imgs_in_cluster)))
    #print(x_test[imgs_in_cluster].shape)
    #plt.subplot(2,1,1)
    #plt.imshow(x_test[imgs_in_cluster])
    #plt.show()
    #plt.savefig("result.jpg")








test_cbir()
