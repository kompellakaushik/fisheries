
# coding: utf-8

# # Fisheris using data augmentation
# 
# ## Project

# In[1]:

# %matplotlib inline
from __future__ import division
# # import matplotlib.pyplot as plt

import numpy as np
np.random.seed(2016)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
import types    

from keras.models import load_model
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras import __version__ as keras_version
from keras import regularizers
from keras.layers import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU,PReLU
from sklearn.externals import joblib
from sklearn.metrics import average_precision_score,accuracy_score,recall_score,precision_score
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import collections
import pdb

from sklearn.metrics import log_loss,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import linear_model,decomposition
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import scipy
import pickle

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn.linear_model import RidgeCV
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit

from scipy.sparse import coo_matrix, hstack ,vstack

import gc

print(gc.collect())

import xgboost as xgb
from keras import applications
import os


# deep models
from keras.layers import Input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


import copy
import datetime
import json
import multiprocessing as mltproc


# In[ ]:




# In[2]:

ResNet50.name='ResNet50'
VGG16.name='VGG16'
VGG19.name='VGG19'
InceptionV3.name='InceptionV3'


# In[3]:

Listallimages=[]
ALB_path='inputdata/train/ALB'
BET_path='inputdata/train/BET'
DOL_path='inputdata/train/DOL'
LAG_path='inputdata/train/LAG'
SHARK_path='inputdata/train/SHARK'
YFT_path='inputdata/train/YFT'
NoF_path = 'inputdata/train/NoF'
OTHER_path = 'inputdata/train/OTHER'


images=os.listdir(ALB_path)
ALB_images=[os.path.join('inputdata/train/ALB',ff) for ff in images if '.jpg' in ff ]


images=os.listdir(BET_path)
BET_images=[os.path.join('inputdata/train/BET',ff) for ff in images if '.jpg' in ff ]

images=os.listdir(DOL_path)
DOL_images=[os.path.join('inputdata/train/DOL',ff) for ff in images if '.jpg' in ff ]

images=os.listdir(LAG_path)
LAG_images=[os.path.join('inputdata/train/LAG',ff) for ff in images if '.jpg' in ff ]

images=os.listdir(SHARK_path)
SHARK_images=[os.path.join('inputdata/train/SHARK',ff) for ff in images if '.jpg' in ff ]

images=os.listdir(YFT_path)
YFT_images=[os.path.join('inputdata/train/YFT',ff) for ff in images if '.jpg' in ff ]

images=os.listdir(NoF_path)
NoF_images=[os.path.join('inputdata/train/NoF',ff) for ff in images if '.jpg' in ff ]

images=os.listdir(OTHER_path)
OTHER_images=[os.path.join('inputdata/train/OTHER',ff) for ff in images if '.jpg' in ff ]

Allimages={'ALB': ALB_images,
           'BET': BET_images,
           'DOL': DOL_images,
           'LAG': LAG_images,
           'SHARK': SHARK_images,
           'YFT': YFT_images,
           'NoF': NoF_images,
           'OTHER': OTHER_images}



# Now getting all the test data
testimages=[os.path.join('inputdata/test_stg1',ff) for ff in os.listdir('inputdata/test_stg1') if '.jpg' in ff ]+             [os.path.join('inputdata/test_stg2',ff) for ff in os.listdir('inputdata/test_stg2') if '.jpg' in ff ]



# In[ ]:




# In[4]:

# All the Neural Network models to be run
class NNmodels(object):
    def __init__(self):
        self.batch_size = 500
        self.nb_epoch = 100
        self.random_state = 51
        self.listofmodels=['NNLinear-1h-500','NNSigmoid-1h-500','NNtanh-1h-500','NNRelu-1h-500','NNLeakyRelu-1h-500']
        self.input_dim=None
        
    def __iter__(self):
        i=0
        # Single Layer
        for l2reg in [0]:
            for N1layer in [50,500]:
                for dropout in np.arange(0,1,0.2):
                    i=i+1
#                     if i>5:
#                         raise StopIteration
                        
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("linear"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNlinear_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'linear'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("sigmoid"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNsigmoid_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'sigmoid'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("tanh"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNtanh_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'tanh'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("relu"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNrelu_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'relu'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(LeakyReLU(alpha = 0.1))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNLeakyReLU_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'LeakyReLU'],model)
        
        
        # Single Layer
        for l2reg in np.linspace(0,0.1,5):
            for N1layer in [50,500]:
                for dropout in [0]:
                    i=i+1
#                     if i>5:
#                         raise StopIteration
                        
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("linear"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNlinear_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'linear'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("sigmoid"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNsigmoid_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'sigmoid'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("tanh"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNtanh_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'tanh'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(Activation("relu"))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNrelu_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'relu'],model)
                    
                    #----------------------------------------
                    model = Sequential()
                    model.add(Dense(output_dim=N1layer, input_dim=self.input_dim,kernel_regularizer=regularizers.l2(l2reg),))
                    model.add(LeakyReLU(alpha = 0.1))
                    model.add(Dropout(dropout))

                    model.add(Dense(output_dim=8))
                    model.add(Activation("softmax"))

                    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.3, nesterov=True)
                    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    name='NNLeakyReLU_%d_%02.4f_%02.4f'%(N1layer,l2reg,dropout)
                    
                    yield (name,[N1layer,l2reg,dropout,'LeakyReLU'],model)
                    
              


# In[5]:

# Extract feature and train on different models
featureExtractors=[
                     {'name':'ResNet50','model':ResNet50(weights='imagenet')},
                     {'name':'VGG16','model':VGG16(weights='imagenet', include_top=True)},
                     {'name':'VGG19','model':VGG19(weights='imagenet', include_top=True)},
                     {'name':'InceptionV3','model':InceptionV3(input_tensor=Input(shape=(224, 224, 3)) ,weights='imagenet', include_top=True)}
                    ]


# In[6]:



class DeepFeatureExtractClassify(object):
    
    def __init__(self,Allimages=[],testimages=[],
                     featureExtractors=featureExtractors,
                     TrainImageMean='TrainImageMean.npz',
                     ModifiedTestData='ModifiedTestData.npz',
                     MeanRemovedData='MeanRemovedData.npz',
                     ExtractedFeatures='ExtractedFeatures.npz'
                        ):
        
        self.TrainImageMean=TrainImageMean
        self.ModifiedTestData=ModifiedTestData
        self.MeanRemovedData=MeanRemovedData
        self.ExtractedFeatures=ExtractedFeatures
        
        self.featureExtractors=featureExtractors
        self.Allimages=Allimages
        self.testimages=testimages
        data=np.load('imagenetlabels.npz')
        self.ImageNetLabels=data['labels']
        
        self.NNmodels=NNmodels()
        gc.collect()
        
#     def __del__(self):
#         del self.Xmean
#         del self.Xext
        
    def GetImagesMean(self):
        if not os.path.isfile(self.TrainImageMean):

            mean_R = []
            mean_G = []
            mean_B = []

            #loading images
            for imageset in self.Allimages.keys():
                print imageset
                for img_path in Allimages[imageset]:
                    img = image.load_img(img_path, target_size=(224, 224,3))

                    #converting images to arrays
                    x = image.img_to_array(img)

                    #finding the mean image
                    a = np.mean(x[:,:,0])
                    mean_R = np.append(mean_R,a)
                    b = np.mean(x[:,:,1])
                    mean_G = np.append(mean_G,b)
                    c = np.mean(x[:,:,2])
                    mean_B = np.append(mean_B,c)

            #Mean Image
            self.I_R = np.mean(mean_R)
            self.I_G = np.mean(mean_G)
            self.I_B = np.mean(mean_B)
            print (self.I_R,self.I_G,self.I_B)
            np.savez(self.TrainImageMean,I_R=self.I_R,I_G=self.I_G,I_B=self.I_B)
        else:
            print "loading saved mean data"
            data=np.load(self.TrainImageMean)
            self.I_R=data['I_R']
            self.I_G=data['I_G']
            self.I_B=data['I_B']
            print (self.I_R,self.I_G,self.I_B)
    
    def SetTestData(self):
        if not os.path.isfile(self.ModifiedTestData):
            self.testfilenames=None
            self.Xtestsubmit=None
            for i,img_path in enumerate(self.testimages):
                print "Mean removing for test image ",str(i)," of ", str( len(self.testimages) ),"\r",
                img = image.load_img(img_path, target_size=(224, 224,3))
                x = image.img_to_array(img)
                x[:,:,0] = x[:,:,0] - self.I_R
                x[:,:,1] = x[:,:,1] - self.I_G
                x[:,:,2] = x[:,:,2] - self.I_B
                x = np.expand_dims(x, axis=0)

                if self.Xtestsubmit is None:
                    self.Xtestsubmit=x
                    self.testfilenames=np.array([img_path])
                else:
                    self.Xtestsubmit=np.vstack((self.Xtestsubmit,x) )
                    self.testfilenames=np.hstack((self.testfilenames,np.array([img_path])) )
            np.savez(self.ModifiedTestData,Xtestsubmit=self.Xtestsubmit,testfilenames=self.testfilenames)
        else:
            print "loading saved test data"
            data=np.load(self.ModifiedTestData)
            self.Xtestsubmit=data['Xtestsubmit']
            self.testfilenames=data['testfilenames']
            
    def SetMeanRemovedData(self):
        if not os.path.isfile(self.MeanRemovedData):
            datagen = ImageDataGenerator(
                    rotation_range=90,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')
            

            print "Removing mean from images"
            Xmean=None
            Y=None
            for imageset in self.Allimages.keys():
                print imageset
                for img_path in Allimages[imageset]:
                    img = image.load_img(img_path, target_size=(224, 224,3))

                    x = image.img_to_array(img)
                    x[:,:,0] = x[:,:,0] - self.I_R
                    x[:,:,1] = x[:,:,1] - self.I_G
                    x[:,:,2] = x[:,:,2] - self.I_B

                    x = np.expand_dims(x, axis=0)
                    if Xmean is None:
                        Xmean=x
                        Y=np.array([imageset])
                    else:
                        Xmean=np.vstack((Xmean,x) )
                        Y=np.vstack((Y,np.array([imageset])) )
            
            try:
                i=0
                self.Xmean=Xmean
                self.Y=Y
                for batch in datagen.flow(Xmean,Y, batch_size=12):
                    i=i+1
                    if i>100:
                        break
                    self.Xmean=np.vstack((self.Xmean,batch[0]) )
                    self.Y=np.vstack((self.Y, batch[1] ) )
            except:
                pdb.set_trace()
            
                
            self.Labels=np.unique(self.Y)
            Y=self.Y
            for ind,l in enumerate(self.Labels):
                Y[np.argwhere(Y==l)]=ind
            self.Ybinary=np_utils.to_categorical(Y)
            np.savez(self.MeanRemovedData,Xmean=self.Xmean,Y=self.Y,Labels=self.Labels,Ybinary=self.Ybinary)
        else:
            print "Loading mean removed data"
            data=np.load(self.MeanRemovedData)
            self.Xmean=data['Xmean']
            self.Y=data['Y']
            self.Y=self.Y.astype(int)
                    
            self.Labels=data['Labels']
            self.Ybinary=data['Ybinary']
            
    def ExtractFeatures(self,rerun=False):
        if not os.path.isfile(self.ExtractedFeatures) or rerun:
#             self.SetMeanRemovedData()
            self.Xext={}
            for model in featureExtractors:
                print "extracting features using deep covnet ",model['name']
                X=None
                for i in range( self.Xmean.shape[0] ):
                    print "Working on image ... "+str(i)+" of "+ str(self.Xmean.shape[0]) +"\r",
                    x = np.expand_dims(self.Xmean[i], axis=0)
                    preds = model['model'].predict(x)
                    if X is None:
                        X=preds[0]
                    else:
                        X=np.vstack( (X,preds[0]) )

                self.Xext[model['name'] ]=X
                print "\nDone\n"
                
            np.savez(self.ExtractedFeatures,Xext=self.Xext)
        else:
            print "Loading Extracted Features"
            data=np.load(self.ExtractedFeatures)
            self.Xext=data['Xext'][()]
    
    
    def ExtractFeatures_parallel(self,rerun=False):
        if not os.path.isfile(self.ExtractedFeatures) or rerun:
#             self.SetMeanRemovedData()
            self.Xext={}
            for model in featureExtractors:
                print "extracting features in parallel using deep covnet ",model['name']
                X=None
                for i in range( self.Xmean.shape[0] ):
                    print "Working on image ... "+str(i)+" of "+ str(self.Xmean.shape[0]) +"\r",
                    x = np.expand_dims(self.Xmean[i], axis=0)
                    preds = model['model'].predict(x)
                    if X is None:
                        X=preds[0]
                    else:
                        X=np.vstack( (X,preds[0]) )

                self.Xext[model['name'] ]=X
                print "\nDone\n"
                
            np.savez(self.ExtractedFeatures,Xext=self.Xext)
        else:
            print "Loading Extracted Features"
            data=np.load(self.ExtractedFeatures)
            self.Xext=data['Xext'][()]
            
            
    def GetSplitTrain(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=0.33, random_state=41)
        return (X_train, X_test, y_train, y_test)
    
    def RunNNmodels_parallel(self,featuremodel,X_train, X_test, y_train, y_test):
        self.NNmodels.input_dim=X_train.shape[1]
        
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        
        batches=[]
        for name,paras,model in self.NNmodels:
            
            if name in self.Results[featuremodel['name']].keys():
                if 'clfs' in self.Results[featuremodel['name']][name].keys():
                    if len(self.Results[featuremodel['name']][name]['clfs'])==1:
                        continue
            
                    
            batches.append((name,paras,model) )
        
        def parallelruns(i,j):
            print i,j
            for name,paras,model in batches[i:j+1]:
                print "\n\n<<<<<<<< ++  >>>>>>>>>>>"
                print "NN model :: ",name
                model.fit(X_train, np_utils.to_categorical(y_train), batch_size=self.NNmodels.batch_size, 
                          nb_epoch=self.NNmodels.nb_epoch,verbose=0, validation_split=0.3)
                score = model.evaluate(X_test,np_utils.to_categorical(y_test), verbose=1)
                print ""
                print '\n\n Test score:', score

                self.Results[featuremodel['name']][name]={ 'clfs':[model],'paras':[paras] }

                dumpname=os.path.join(self.modelspath,  featuremodel['name']+'_'+name+'.h5')
                model.save(dumpname)
                self.ResultsSave[featuremodel['name']]['XGBOOST']={ 'clfs':[dumpname],'paras':[paras]  }
        

        P=[]
        for chk in chunks(range(len(batches)),100):
            P.append( mltproc.Process(target=parallelruns, args=(chk[0],chk[-1],)) )

        for p in P:
            p.start()
        for p in P:
            p.join()

    def RunNNmodels(self,featuremodel,X_train, X_test, y_train, y_test):
        self.NNmodels.input_dim=X_train.shape[1]
        
        i=1
        for name,paras,model in self.NNmodels:
            if name in self.Results[featuremodel['name']].keys():
                if 'clfs' in self.Results[featuremodel['name']][name].keys():
                    if len(self.Results[featuremodel['name']][name]['clfs'])==1:
                        continue
                    
            print "\n\n<<<<<<<< ++  >>>>>>>>>>>"
            print "NN model :: ",name
            model.fit(X_train, np_utils.to_categorical(y_train), batch_size=self.NNmodels.batch_size, 
                      nb_epoch=self.NNmodels.nb_epoch,verbose=0, validation_split=0.3)
            score = model.evaluate(X_test,np_utils.to_categorical(y_test), verbose=1)
            print ""
            print '\n\n Test score:', score

            self.Results[featuremodel['name']][name]={ 'clfs':[model],'paras':[paras] }

            dumpname= featuremodel['name']+'_'+name+'.h5'
            model.save(os.path.join(self.modelspath,  dumpname))
            self.ResultsSave[featuremodel['name']][name]={ 'clfs':[dumpname],'paras':[paras]  }

            if i%10==0:
                with open(self.Resultsfile,'w') as F:
                    json.dump(self.ResultsSave,F, indent=4, separators=(',', ': '))
            i=i+1
            
    def RunXGBoost(self,featuremodel,X_train, X_test, y_train, y_test):
        
        if 'XGBOOST' in self.Results[featuremodel['name']].keys():
            if 'clfs' in self.Results[featuremodel['name']]['XGBOOST'].keys():
                if len(self.Results[featuremodel['name']]['XGBOOST']['clfs'])==1:
                    return
                    
        dtrain = xgb.DMatrix(X_train, label=y_train.reshape(-1,1))
        dtest = xgb.DMatrix(X_test)

        param = {'max_depth':100, 'eta':0.02, 'silent':1, 'objective':'multi:softmax','num_class':8 }
        param['nthread'] = 6
        param['eval_metric'] = 'mlogloss'
        param['subsample'] = 0.7
        param['colsample_bytree']= 0.7
        param['min_child_weight'] = 0
        param['booster'] = "gblinear"

        watchlist  = [(dtrain,'train')]
        num_round = 300
        early_stopping_rounds=10

        clf_xgb = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds=early_stopping_rounds,verbose_eval = False)

        Ypred = clf_xgb.predict(dtest)
        # y_test=np_utils.to_categorical(y_test)

        print "\n++ Accuracy Score ++\n"
        print metrics.accuracy_score(y_test,Ypred)
        print "\n++ Classification report ++\n"
        print metrics.classification_report(y_test,Ypred)
        print "\n++ Confusion Matrix ++\n"
        print '\x1b[1;31m'+ str(metrics.confusion_matrix(y_test,Ypred) )+'\x1b[0m'
        
        self.Results[featuremodel['name']]['XGBoost']={ 'clfs':[clf_xgb] }
        
        dumpname= featuremodel['name']+'_'+'XGBOOST.model'
        joblib.dump(clf_xgb, os.path.join(self.modelspath, dumpname) )
#         clf_xgb.save_model( os.path.join(self.modelspath, dumpname) )
        self.ResultsSave[featuremodel['name']]['XGBOOST']={ 'clfs':[dumpname] }
    
        with open(self.Resultsfile,'w') as F:
            json.dump(self.ResultsSave,F, indent=4, separators=(',', ': '))            
    
                    
    def RunClassifiers(self,modeldir=None,skipdone=True):
        """
        Run all the classifiers with cross validation and grid search
        
        """

        if modeldir==None:
            self.version=str( datetime.datetime.now()).split('.')[0]
            if not os.path.isdir('models_'+self.version):
                os.mkdir('models_'+self.version)
            self.modelspath='models_'+self.version
        else:
            self.modelspath=modeldir
    
        if skipdone==True:
            self.LoadModels(modeldir=modeldir)
        else:
            self.Results={}
            self.ResultsSave={}
         
        self.Resultsfile=os.path.join(self.modelspath, 'Results.json')
        
        
            
        for featuremodel in featureExtractors:
            print "running classification on features extracted by ", featuremodel['name']

            X_train, X_test, y_train, y_test=self.GetSplitTrain(self.Xext[featuremodel['name'] ],self.Y)

            ModelParaGrid=[{'name':'LinearSVC','model':LinearSVC(C=1),'para':{'C': [1, 10, 100, 1000,10000]}},
                           {'name':'SVC','model':SVC(C=1.0, kernel='linear', max_iter=1e5,verbose=True, probability=True, shrinking=True) , 'para':{'C': [1, 10, 100, 1000,10000]} },
                            {'name':'LogisticRegression','model':LogisticRegression(C=1e1,n_jobs=5,verbose=1),
                             'para':{'C': [1, 10, 100, 1000,10000]}},
                            {'name':'RandomForestClassifier','model':RandomForestClassifier(n_estimators=15,
                                                                n_jobs=5,max_depth=40,max_features=60),
                             'para':{'n_estimators':[10,100,250,500],'max_depth':[10,100,250,500], 'max_features': [30,40,50,60] }},
                            ]
            if featuremodel['name'] not in self.Results.keys():
                self.Results[featuremodel['name']]={}
                self.ResultsSave[featuremodel['name']]={}
            
            print "#####################################################################################"
            print "--------------------  "+ featuremodel['name'] +"  -----------------------------------"
            print "#####################################################################################"


            for M in ModelParaGrid:
                scores = ['precision', 'recall']
                print "************ " + M['name'] + "******************"
                
                if M['name'] not in self.Results[featuremodel['name']].keys():
                    self.Results[featuremodel['name']][M['name']]={'clfs':[] }
                    self.ResultsSave[featuremodel['name']][M['name']]={'clfs':[]}

                if len( self.Results[featuremodel['name']][M['name']]['clfs'] )==0:
                    for score in scores:
                        print "# Tuning hyper-parameters for %s" % score
                        print ""

                        clf = GridSearchCV(M['model'], M['para'], cv=3,
                                           scoring='%s_macro' % score,n_jobs=5)
                        clf.fit(X_train,  y_train.reshape(1,-1)[0])

                        print "Best parameters set found on development set:"
                        print 
                        print clf.best_params_
                        print 
                        print "Grid scores on development set:"
                        print
                        means = clf.cv_results_['mean_test_score']
                        stds = clf.cv_results_['std_test_score']
                        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                            print "%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params) 
                        print

                        print "Detailed classification report:"
                        print
                        print "The model is trained on the full development set."
                        print "The scores are computed on the full evaluation set."
                        print
                        y_true, y_pred = y_test, clf.predict(X_test)
                        print classification_report(y_true, y_pred)

                        print
                        self.Results[featuremodel['name']][M['name']]['clfs'].append( clf )

                        dumpname= featuremodel['name']+'_'+M['name']+'_'+score+'.pkl'
                        joblib.dump(clf, os.path.join(self.modelspath, dumpname) )
                        self.ResultsSave[featuremodel['name']][M['name']]['clfs'].append(dumpname)
                        
                        with open(self.Resultsfile,'w') as F:
                            json.dump(self.ResultsSave,F, indent=4, separators=(',', ': '))
                
            # XGBoost
            print '-----------------  XGBOOST - tree  ------------------------------'
            self.RunXGBoost(featuremodel,X_train, X_test, y_train, y_test)

            # Running keras models on extracted features
#             print "----------------- Keras NN models ----------------------------------"
#             self.RunNNmodels(featuremodel,X_train, X_test, y_train, y_test)
#             self.RunNNmodels_parallel(featuremodel,X_train, X_test, y_train, y_test)
            
        
            with open(self.Resultsfile,'w') as F:
                json.dump(self.ResultsSave,F, indent=4, separators=(',', ': '))
            
    def LoadModels(self,modeldir=None):
        if modeldir is None:
            modeldirectories=[ff for ff in os.listdir('.') if os.path.isdir(ff) and 'model' in ff]
        
        print modeldir
        self.Resultsfile=os.path.join( modeldir , 'Results.json')
        if os.path.isfile(self.Resultsfile):
            with open(self.Resultsfile,'r') as F:
                self.ResultsSave=json.load(F)
            
            print "loading saved models from the saved json"
            
            self.Results=copy.deepcopy(self.ResultsSave)
            return 
        
            if not hasattr(self,'Results'):
                self.Results={}
    
            for FM in self.ResultsSave.keys():
                if FM not in self.Results.keys():
                    self.Results[FM]={}

                for clftype in self.ResultsSave[FM].keys():
                    if clftype not in self.Results[FM].keys():
                        self.Results[FM][clftype]={'clfs':[]}
                    if len( self.Results[FM][clftype]['clfs'] )==0:
                        print "Loading model for ",FM," ",clftype,'\r',
                        
                        for i in range( len(self.ResultsSave[FM][clftype]['clfs']) ):
                            ff=self.ResultsSave[FM][clftype]['clfs'][i]
                            if '.pkl' in ff:
                                clf = joblib.load(os.path.join(modeldir,ff) )
                            elif '.model' in ff:
    #                             clf = xgb.Booster({'nthread':4}) #init model
    #                             clf.load_model(os.path.join(modeldir,ff)) # load data
                                clf = joblib.load(os.path.join(modeldir,ff))
                                def predict_custom(self,X):
                                    dtest = xgb.DMatrix(X)
                                    y=self.predict(dtest)
                                    return np_utils.to_categorical(y.reshape(-1,1))
                                clf.predict_custom=types.MethodType( predict_custom, clf )

                            elif '.h5' in ff:
                                clf = load_model(os.path.join(modeldir,ff))
                            else:
                                print "Model name not knwon"
                                clf=None
                            self.Results[FM][clftype]['clfs'].append(clf)
        else:
            self.Results={}
            self.ResultsSave={}
               
            # construct from the available
            print "Results file not there"
            os.listdir(modeldir)
            i=0
            for model in os.listdir(modeldir):
                FE=model.split('_')[0]
                clfstr=".".join( "_".join(model.split('_')[1:]).split('.')[:-1] )
                clfstr=clfstr.replace('_recall','').replace('_precision','')
                i=i+1
                print "Loading model for ",model," ",i," of ",len(os.listdir(modeldir)),'\r',
                
                if FE not in self.Results.keys():
                    self.Results[FE]={}
                    self.ResultsSave[FE]={}
                if clfstr not in self.Results[FE].keys():
                    self.Results[FE][clfstr]={'clfs':[]}
                    self.ResultsSave[FE][clfstr]={'clfs':[]}

                if '.pkl' in model:
                    clf = joblib.load(os.path.join(modeldir,model))
                elif '.model' in model:
#                     clf = xgb.Booster({'nthread':4}) #init model
#                     clf.load_model(os.path.join(modeldir,model)) # load data
                    clf = joblib.load(os.path.join(modeldir,model))
                    def predict_custom(self,X):
                        dtest = xgb.DMatrix(X)
                        y=self.predict(dtest)
                        return np_utils.to_categorical(y.reshape(-1,1))
                    clf.predict_custom=types.MethodType( predict_custom, clf )
                    
                elif '.h5' in model:
#                     print os.path.join(modeldir,model)
                    clf = load_model(os.path.join(modeldir,model))
                else:
                    print "Model name not knwon"
                    clf=None
                self.Results[FE][clfstr]['clfs'].append(clf)
                self.ResultsSave[FE][clfstr]['clfs'].append(model)
                
            with open(self.Resultsfile,'w') as F:
                json.dump(self.ResultsSave,F, indent=4, separators=(',', ': '))

    def EvalPerformance(self,savetag=''):
        self.PerFormance=[]
        for N in range(25):

            print "Generating random test data ",N
            train_idx, test_idx, _, _ = train_test_split(np.arange(self.Xmean.shape[0]), np.arange(self.Xmean.shape[0]), test_size=0.33, random_state=41)

            print "Evaluating performance"
            
            for FM in self.Results.keys():
                print "--------- ",FM, " -----------"

                Xtest=self.Xext[ FM ][test_idx,:]
                Ytest=self.Y[test_idx]

                for clftype in self.Results[FM].keys():
                    for i in range( len(self.Results[FM][clftype]['clfs']) ):
                        print "\n\n"+FM+' '+clftype+' '+str(i)

                        clf=self.Results[FM][clftype]['clfs'][i]
                        if hasattr(clf,'predict_custom'):
                            y_pred=clf.predict_custom(Xtest)
                        elif hasattr(clf,'predict_proba'):
                            y_pred=clf.predict_proba(Xtest)
                        else:
                            y_pred=clf.predict(Xtest)
                            try:
                                y_pred.shape[1]
                            except:
                                y_pred=np_utils.to_categorical(y_pred.reshape(-1,1))

                        logloss=log_loss(Ytest, y_pred, eps=1e-15, normalize=True)
                        avgprec= average_precision_score(np_utils.to_categorical(Ytest), y_pred)
                        acc= accuracy_score(Ytest, np.argmax(y_pred,axis=1) )
                        recallscore= recall_score(Ytest, np.argmax(y_pred,axis=1),average='micro' )
                        precisionscore= precision_score(Ytest, np.argmax(y_pred,axis=1),average='micro' )

                        self.PerFormance.append( {'FeatureExtractor': FM, 
                                                  'Classifier':clftype+'_'+str(i), 
                                                  'log_loss' : logloss,
                                                  'acc'   : acc,
                                                  'avgprec':avgprec,
                                                  'recallscore':recallscore,
                                                  'precisionscore':precisionscore,
                                                 } )
                        print "\r",

        df=pd.DataFrame(self.PerFormance)
        df.sort_values(by='log_loss',ascending=True,inplace=True)

        print df[['FeatureExtractor','Classifier','log_loss','acc']]

        df.to_csv('Performance_'+savetag+'.csv')
        df.to_hdf('Performance_'+savetag+'.h5','table')

        
#     def GenerateSubmission(self,top=10,modeldir='',savetag=''):
#         self.LoadModels(modeldir=modeldir)
        
#         df=pd.read_hdf('Performance_'+savetag+'.h5','table')
#         df.sort_values(by='log_loss',ascending=True,inplace=True)
#         df.index=range(len(df))
        
#         for ind in df.index:
#             self.ResultsSave[  ]


# In[7]:

# Run on augmented data
DFEC=DeepFeatureExtractClassify(Allimages=Allimages,testimages=testimages,
                                        TrainImageMean='TrainImageMean.npz',
                                        ModifiedTestData='MeanRemovedTestData.npz',
                                        MeanRemovedData='MeanRemovedData_aug.npz',
                                        ExtractedFeatures='ExtractedFeatures_aug.npz')
DFEC.GetImagesMean()
# #DFEC.SetTestData()
DFEC.SetMeanRemovedData()
DFEC.ExtractFeatures(rerun=False)
DFEC.RunClassifiers(modeldir='models_2017-04-20 14:34:12',skipdone=True)
DFEC.EvalPerformance(savetag='aug')
df=pd.read_hdf('Performance_aug.h5','table')
df


# In[8]:

# # Run un-augmented data
# DFEC=DeepFeatureExtractClassify(Allimages=Allimages,testimages=testimages,
#                                         TrainImageMean='TrainImageMean.npz',
#                                         ModifiedTestData='MeanRemovedTestData.npz',
#                                         MeanRemovedData='MeanRemovedData_noaug.npz',
#                                         ExtractedFeatures='ExtractedFeatures_noaug.npz')
# DFEC.GetImagesMean()
# DFEC.SetMeanRemovedData()
# DFEC.ExtractFeatures(rerun=False)
# DFEC.RunClassifiers(modeldir=None,skipdone=False)
# DFEC.EvalPerformance(savetag='noaug')
# df=pd.read_hdf('Performance_noaug.h5','table')
# df


# In[10]:

df.shape


# In[11]:

a=np.array([1,2,3])
np.savez('test.npz',a=a)


# In[ ]:




# In[ ]:




# In[ ]:



