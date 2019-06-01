



from skimage import img_as_float
import pickle
import os
import numpy as np
import collections
import cv2





class FeaturesIndex:
    def __init__(self):
        self.feat_index = collections.defaultdict(list)
    def register_cluster(self, cluster, cluster_vects ):
        for i in range(len(cluster_vects)):
                self.feat_index[cluster].extend(cluster_vects)
    def save(self):
        path = os.path.join(os.getcwd(), "feature_index.pkl")
        with open(path, "wb") as f:
            save_index ={
            "feat_index" :dict(self.feat_index)
            }
            pickle.dump(save_index, f, pickle.HIGHEST_PROTOCOL)


    def load(self):
        path = os.path.join(os.getcwd(), "feature_index.pkl")
        with open(path, "rb") as f:
            load_index = pickle.load(f)
            for k in load_index:
                delattr(self, k)
                setattr(self,k, load_index[k] )




def load_dataset(data_dir, train=True, as_grey=False, shuffle=True):

    y = []
    X = []
    class_names = []

    if train:
        data_dir = os.path.join(data_dir, 'dataset-train')
    else:
        data_dir = os.path.join(data_dir, 'dataset-test')

    for i, cls in enumerate(sorted(os.listdir(data_dir))):
        for img_file in os.listdir(os.path.join(data_dir, cls)):
            img_path = os.path.join(data_dir, cls, img_file)
            img = img_as_float( cv2.cvtColor(cv2.resize(cv2.imread(img_path), (124,124)),cv2.COLOR_BGR2GRAY))
            #img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
            X.append(img)
            y.append(i)
        class_names.append(cls)

    # Convert list of imgs and labels into array
    X = np.array(X)
    y = np.array(y)

    if shuffle:
        idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]

    return np.array(X), np.array(y), class_names
