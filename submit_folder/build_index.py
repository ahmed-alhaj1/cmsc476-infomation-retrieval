import os
from extractor_models import De_Conv_Autoencoder, plot_sample_imgs, plot_encoded_imgs
import numpy as np
from skimage import io
from skimage import img_as_float
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.externals import joblib
from load_data import load_dataset, FeaturesIndex
import collections
from sklearn.cluster import KMeans
import cv2
import h5py
from matplotlib import pyplot as plt






'''
    Extract feature and index the image
'''

def extract_feature(deep_model, x_test):
    feats = deep_model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
    print(feats)
    features_vects = []
    for feat in feats:
        feat_vector = feat[0] / np.linalg.norm(feat[0])
        features_vects.append( feat_vector.flatten())
    return features_vects


def trigger_clustering(labels, features_stack, count):
    print("trigger_clustering")
    n_clusters = 100
    #kmeans = 0
    #kmeans = Kmeans(n_clusters = n_clusters)
    #n_samples , HI, WI = features_stack.shape
    #d2_features_stack = features_stack.reshape((n_samples, HI*WI))
    #labeled_feat_stack = np.vstack((labels, features_stack))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0).fit(np.array(features_stack))
    return kmeans

    #kmeans_fit = keams.fit_predict(desc_stack)
    #feat_hist = np.array([np.zeros(n_clusters) for i in range(len(feat_vect_list))])

def Index_histogram(km_model, n_clusters):
    print("building index histogram")
    index = FeaturesIndex()

    for i in range(n_clusters):
        feat_vects = np.where(km_model.labels_ == i)[0]
        print("feat_vector --> ", feat_vects)
        index.register_cluster(i, feat_vects)
        index.save()
    print("index stuff")
    for k in index.feat_index:
        print(k, " --> ", index.feat_index[k])

    file_name = os.path.join(os.getcwd(), "kmeans_data.pkl")
    joblib.dump(km_model, file_name)








def build_index():

    de_conv_encoder = De_Conv_Autoencoder()
    de_conv_encoder.build_auto_encoder()
    de_conv_encoder.compile()
    de_conv_encoder.load()

    x_train, x_test, labels = load_dataset(os.getcwd(), False)
    feat_vect_list = []
    #for i in range(len(x_train)):
    feat_vect_list = extract_feature(de_conv_encoder, x_train)


    n_clusters = 100
    km_model  = trigger_clustering(labels, np.array(feat_vect_list), count= n_clusters)
    Index_histogram(km_model, n_clusters)

build_index()

    #vect_stack = feat_vect_list[0]
    #for i in feat_vect_list[1:]:
    #vect_stack = np.vstack((vect_stack, i))
