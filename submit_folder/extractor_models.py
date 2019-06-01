from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import numpy as np
from matplotlib import pyplot  as plt
import pickle
import numpy as np
from numpy import linalg as LA



class De_Conv_Autoencoder():
    def __init__(self, features_patch=None, label_path= None):
        self.features_patch = features_patch
        self.labels_path = label_path
        self.auto_encoder =None
        self.dp_encoder = None
        self.decoder = None

    def build_auto_encoder(self, encoding_dim = 32, img = None):

        encoding_dim = 32
        input_img = Input(shape=(124, 124, 1))  # adapt this if using `channels_first` image data format

        #x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)



        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        #x = UpSampling2D((2, 2))(x)
        #x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        decoded  = UpSampling2D((2, 2))(x)



        self.dp_encoder = Model(input_img, decoded)


        return  self.dp_encoder.summary()

    def compile(self):
        self.dp_encoder.compile(optimizer='adadelta', loss = 'binary_crossentropy')

    def feed_data(self,x_train_noise, x_train, x_test_noise, x_test):

        x_test  = x_test.astype('float32')/255


        ####################################################
        ####################################################

        #x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        #x_test_noisy = np.clip(x_test_noisy, 0., 1.)


        self.dp_encoder.fit(x_train_noise, x_train, epochs= 100, batch_size=128, shuffle=True, validation_data=(x_test_noise, x_test),verbose=2)


    def add_noise(self, x_train, x_test):
        #t0   = time.time()
        #x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')/255
        x_test = x_test.astype('float32') /255

        x_train = np.reshape(x_train,(len(x_train) , 124, 124,1))
        x_test = np.reshape(x_test ,  (len(x_test) , 124, 124,1))

        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        #t1 = time.time()
        return x_train, x_train_noisy, x_test, x_test_noisy

    def predict(self, x_test ):
        try:
            print("try ")
            return self.dp_encoder.predict(x_test)

        except:
            print("except")

            return self.dp_encoder.predict(x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    def save(self):
        self.dp_encoder.save('de_conv_autoencoder.h5')
    def load(self):
        self.dp_encoder.load_weights('de_conv_autoencoder.h5', by_name=False)


##############################################################################################
##############################################################################################
class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = self.model.predict(img)
        norm_feat = features[0]/LA.norm(features[0])
        return norm_feat




def plot_encoded_imgs(encoded_imgs, n =10):
    plt.figure(figsize= (10,4), dpi=100)
    for i in range(n):
        ax= plt.subplot(2,n,i+n+1)
        plt.imshow(encoded_imgs[i].reshape(124,124))
        plt.gray()
        ax.set_axis_off()
def plot_decoded_imgs(decoded_imgs, n =10):
    plt.figure(figsize= (10,2), dpi=100)
    for i in range(n):
        ax = plt.subplot(2,n,i+1+n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.set_axis_off()
def plot_sample_imgs(x_test, n=10):
    plt.figure(figsize=(10,4), dpi=100)
    for i in range(n):
        ax = plt.subplot(2,n,i+n+1)
        plt.imshow(x_test[i].reshape(124,124))
        plt.gray()
        ax.set_axis_off()
