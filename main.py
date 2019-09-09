#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:40:04 2019

@author: truc

run on terminal with command line to save a log file:
    
    python main.py >log/main_base12.txt 2>error/main_base12.err
"""


# coding: utf-8

# In[ ]:


# select a GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[ ]:


#imports 
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import librosa
import librosa.display
import soundfile as sound
import pickle
import copy
import time
import logging

import keras
import tensorflow
from keras.optimizers import SGD, Adadelta
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from DCASE2019_models import model_best2019_base1, model_best2019_base2

from DCASE_training_functions import LR_WarmRestart

from utils.DCASE_plots import plot_confusion_matrix
from utils.log_utils import setup_logging, FancyLogger, Timer
from utils.callbacks import ProgressLoggerCallback, StasherCallback, SnapshotCallback


from mixup_generator import MixupGenerator

import random
random.seed(30)

print ("Random number with seed 30") 
print("Librosa version = ",librosa.__version__)
print("Pysoundfile version = ",sound.__version__)
print("keras version = ",keras.__version__)
print("tensorflow version = ",tensorflow.__version__)


# In[ ]:


#WhichTask = '1a'
WhichTask = '1b'
#WhichTask = '1c'

if WhichTask =='1a':
    DatasetPath = '../TAU-urban-acoustic-scenes-2019-development/'
    TrainFile = DatasetPath + 'evaluation_setup/fold1_train.csv'
    ValFile = DatasetPath + 'evaluation_setup/fold1_evaluate.csv'
    sr = 48000
    num_audio_channels = 2
elif WhichTask =='1b':
    DatasetPath = '/clusterFS/home/user/truc/DCASE2019/task1/datasets/TAU-urban-acoustic-scenes-2019-mobile-development/'
    TrainFile = DatasetPath + 'evaluation_setup/fold1_train.csv'
    ValFile = DatasetPath + 'evaluation_setup/fold1_evaluate.csv'
    sr = 44100
    num_audio_channels = 1
elif WhichTask =='1c':
    DatasetPath = '../Task1c/'
    TrainFile = DatasetPath + 'evaluation_setup/fold1_train.csv'
    sr = 44100
    num_audio_channels = 1
    
SampleDuration = 10

#log-mel spectrogram parameters
NumFreqBins = 128 #256  #128 #
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2) #4)
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))
eps = np.spacing(1) 

#training parameters
max_lr = 0.5
batch_size = 64
num_epochs = 510
mixup_alpha = 0.2
crop_length = 400

# feature using delta setting
delta_delta = False

# flags 
normalization_flag = True


# In[]: Manage model filename

model_names = [
#        'model_best2019_base1.h5',
        'model_best2019_base1_SpeCor_Norm.h5',
#        'model_best2019_base2_SpeCor_Norm.h5'
        ]


#def main():
# In[]: Manage path for storing system outputs
Basepath = '/clusterFS/home/user/truc/DCASE2019/NEW_BASELINE/'
spectrum_path = Basepath + 'system_outputs/task1b/spectrums/'
if not os.path.exists(spectrum_path):
    os.makedirs(spectrum_path)
    
feature_path = Basepath + 'system_outputs/task1b/features/'
if not os.path.exists(feature_path):
    os.makedirs(feature_path)
    
normalization_path = Basepath + 'system_outputs/task1b/normalizations/'
if not os.path.exists(normalization_path):
    os.makedirs(normalization_path)
    
model_path = Basepath + 'system_outputs/task1b/learners/'
if not os.path.exists(model_path):
    os.makedirs(model_path)

result_path = Basepath + 'system_outputs/task1b/recognizers/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
    
log_path = Basepath + 'log/'
if not os.path.exists(log_path):
    os.makedirs(log_path)   
    
## In[]: Manage path for storing system outputs
#Basepath = '/clusterFS/home/user/truc/DCASE2019/NEW_BASELINE/'
#spectrum_path = Basepath + 'system_outputs/task1b_128_431/spectrums/'
#if not os.path.exists(spectrum_path):
#    os.makedirs(spectrum_path)
#    
#feature_path = Basepath + 'system_outputs/task1b_128_431/features/'
#if not os.path.exists(feature_path):
#    os.makedirs(feature_path)
#    
#normalization_path = Basepath + 'system_outputs/task1b_128_431/normalizations/'
#if not os.path.exists(normalization_path):
#    os.makedirs(normalization_path)
#    
#model_path = Basepath + 'system_outputs/task1b_128_431/learners/'
#if not os.path.exists(model_path):
#    os.makedirs(model_path)
#
#result_path = Basepath + 'system_outputs/task1b_128_431/recognizers/'
#if not os.path.exists(result_path):
#    os.makedirs(result_path)
#    
#log_path = Basepath + 'log_128_431/'
#if not os.path.exists(log_path):
#    os.makedirs(log_path)   
# In[ ]:Set loging file
'''
from utils.log_utils import setup_logging, FancyLogger, Timer

using log.line(data='', indent=2) instead of print to save to log file
'''

# Setup logging
setup_logging(
    logging_file=log_path + 'model_best2019_base1.yaml'
    )
# Get logging interface
log = FancyLogger()
timer = Timer()
# Log title
log.title('DCASE2019 / Task1B -- Acoustic scene classification')
log.line()

#log_model = logging.getLogger('tensorflow')
#log.setLevel(logging.DEBUG)


# In[ ]: Manage labels and training and validation sets

'''There are 16530 files in audio but 1030 files unsed for training set
   number of audio files for training set and val set is 15500
'''

#load filenames and labels
dev_train_df = pd.read_csv(TrainFile,sep='\t', encoding='ASCII')
dev_val_df = pd.read_csv(ValFile,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
wavpaths_val = dev_val_df['filename'].tolist()
wavpaths_all = sorted(wavpaths_train + wavpaths_val)

filename_list_train = [os.path.splitext(os.path.basename(filename))[0] for filename in wavpaths_train]
filename_list_val = [os.path.splitext(os.path.basename(filename))[0] for filename in wavpaths_val]
filename_list = sorted(filename_list_train + filename_list_val)

y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values
y_val_labels =  dev_val_df['scene_label'].astype('category').cat.codes.values

ClassNames = np.unique(dev_train_df['scene_label'])
NumClasses = len(ClassNames)

y_train = keras.utils.to_categorical(y_train_labels, NumClasses)
y_val = keras.utils.to_categorical(y_val_labels, NumClasses)


# In[ ]:Do spectrum extraction


# Feature extraction stage
log.section_header('Spectrum Extraction')
timer.start()
if not len(wavpaths_all)==len(filename_list): 
    log.line('Error!!!', indent=2)


spectrum = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list):
    
    audio_filename = DatasetPath + wavpaths_all[i]
    spectrum_filename = spectrum_path + filename + '.cpickle'
    #
    if not os.path.isfile(spectrum_filename):
        log.line('{}/{} {}.wav'.format(i, len(filename_list), filename))
        # Read sound file
        stereo,fs = sound.read(audio_filename,stop=SampleDuration*sr)
        for channel in range(num_audio_channels):
            if len(stereo.shape)==1:
                stereo = np.expand_dims(stereo,-1)
            # Do mel extraction
            spectrum = np.abs(librosa.core.stft(stereo[:,channel], #+ self.eps
                                           n_fft      = NumFFTPoints,
                                           hop_length = HopLength,
                                           window     = 'hann', 
                                           center     = True)
            )
            #---spectrum.shape = (1025,431,1)
    
        ##save feature file
        pickle.dump(spectrum, open(spectrum_filename, "wb" ) )

##print('-----DONE') 
timer.stop()
log.foot(time=timer.elapsed())
       
        
# In[ ]:Calculated spectrum correction coefficients    
        

log.section_header('Spectrum Correction')
timer.start()
log.line('-----Calculated spectrum correction coefficients')  

spectrumcorrection_filename_path = normalization_path + 'spectrumcorrection.cpickle'

if not os.path.isfile(spectrumcorrection_filename_path):  
    ## Processing spectrum correction
    filename_list_a = []
    filename_list_b = []
    filename_list_c = []
    same_filename_list_b = []
    
    # Get filename list of dev B -- preparing spectrum pairs for spectrum correction 
    for filename in filename_list_train:
        if filename[-2:] == '-b':
            # Get feature filename
            spectrum_filename = spectrum_path + filename + '.cpickle'
            filename_list_b.append(spectrum_filename)
            # Get same string in feature filename
            same_filename_list_b.append(filename[:-2])
                    
    # Get filename for dev.A and C which have same filenames of Dev.B    
    for filename in filename_list_train:
        for same_filename_b in same_filename_list_b:
            # Get the same filename for dev C
            if filename[-2:] == '-c' and filename[:-2] == same_filename_b:
                spectrum_filename = spectrum_path + filename + '.cpickle'
                filename_list_c.append(spectrum_filename)
            # Get the same filename for Dev A    
            if filename[-2:] == '-a' and filename[:-2]==same_filename_b:
                spectrum_filename = spectrum_path + filename + '.cpickle'
                filename_list_a.append(spectrum_filename)   
                    
    # Do spectrum correction for data_ref = Dev. A and data_x = Dev. B / C   
    sum_divided_spectrum = None    
    sum_divided_spectrum_ref = None 
    sum_divided_spectrum_ref_1 = None 
    for spectrum_a, spectrum_b, spectrum_c,  in zip(filename_list_a, filename_list_b, filename_list_c):
        # Load feature matrix
        features_a = pickle.load(open(spectrum_a, 'rb'))
        features_b = pickle.load(open(spectrum_b, 'rb'))
        features_c = pickle.load(open(spectrum_c, 'rb'))
        
        # Accumulate of divided spectrum statistics
        data_ref1=features_b
        data_ref2=features_c
        data_x=features_a
        #
        data_ref = np.mean((data_ref1 + data_ref2)/2., axis=1)
        data_ref_ref = (np.mean(data_ref1, axis=1) + np.mean(data_ref2, axis=1))/2.
        data_ref_ref_1 = (data_ref1 + data_ref2)/2.
        divided_spectrum = data_ref/np.mean(data_x, axis=1)
        divided_spectrum_ref = data_ref_ref/np.mean(data_x, axis=1)
        #divided_spectrum_ref_1 = data_ref_ref_1/data_x 
        '''dividing first in for loop and finally doing mean will cause the coefficient in difference
        divided_spectrum_ref_1 = data_ref_ref_1/data_x
    #coefficients_ref_1 = (np.mean((sum_divided_spectrum_ref_1/len(filename_list_a)),axis=1)).reshape(-1, 1)
    #coefficients_ref_2 = (np.mean(sum_divided_spectrum_ref_1,axis=1)/len(filename_list_a)).reshape(-1, 1)
        '''
        #
        if sum_divided_spectrum is None:
            sum_divided_spectrum = divided_spectrum
            sum_divided_spectrum_ref = divided_spectrum_ref
            #sum_divided_spectrum_ref_1 = divided_spectrum_ref_1
        else:
            sum_divided_spectrum += divided_spectrum
            sum_divided_spectrum_ref += divided_spectrum_ref
            #sum_divided_spectrum_ref_1 += divided_spectrum_ref_1
        #
    # Calculate coefficients of spectrum correction for all spectrum pairs
    coefficients = (sum_divided_spectrum/len(filename_list_a)).reshape(-1, 1)
    coefficients_ref = (sum_divided_spectrum_ref/len(filename_list_a)).reshape(-1, 1)
    #coefficients_ref_1 = (np.mean((sum_divided_spectrum_ref_1/len(filename_list_a)),axis=1)).reshape(-1, 1)
    #coefficients_ref_2 = (np.mean(sum_divided_spectrum_ref_1,axis=1)/len(filename_list_a)).reshape(-1, 1)
    if (np.abs(coefficients - coefficients_ref)).all() < 0.00001:
        log.line('Same coefficients')
    ##save feature file
    pickle.dump(coefficients, open(spectrumcorrection_filename_path, 'wb' ) )
         
##print('-----DONE') 
timer.stop()
log.foot(time=timer.elapsed())        


# In[ ]:Do feature extraction


#load spectrums and spectrum coefficients and 
log.section_header('Feature Extraction')
timer.start()
print('-----Do feature extraction')

SC_coefficients = pickle.load(open(spectrumcorrection_filename_path, 'rb'))
print('SC_coefficients =')
print(SC_coefficients)

#mel_feature = np.zeros((NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list):
    
    spectrum_filename = spectrum_path + filename + '.cpickle'
    feature_filename = feature_path + filename + '.cpickle'
    #
    if not os.path.isfile(feature_filename):
        print('{}/{} {}.wav'.format(i,len(filename_list),filename))  
        # Read spectrum coefficients
        spectrum = pickle.load(open(spectrum_filename, 'rb'))
        spectrum = copy.deepcopy(spectrum)
        corrected_spectrum = spectrum * SC_coefficients
        
        ## display spectrum
#        import matplotlib.pyplot as plt
#        plt.figure()
#        librosa.display.specshow(corrected_spectrum, y_axis='mel', x_axis='time')
#        plt.colorbar(format='%+2.0f dB')
#        plt.title('Spectrum')
#        plt.tight_layout()
#        plt.show()
            
        # mel basis   mel_basis.shape=(128, 1025)
        mel_basis = librosa.filters.mel(sr    = sr,
                                       n_fft  = NumFFTPoints,
                                       n_mels = NumFreqBins,
                                       fmin   = 0.0,
                                       fmax   = sr/2,
                                       htk    = True,
                                       norm   = None)
                                     
        mel_feature = np.dot(mel_basis, corrected_spectrum)
        log_mel_feature = np.log(np.abs(mel_feature + eps))
        #---mel_feature.shape = (128, 431)
        
#        ## display spectrogram
#        import matplotlib.pyplot as plt
#        plt.figure()
#        librosa.display.specshow(log_mel_feature, y_axis='mel', x_axis='time')
#        plt.colorbar(format='%+2.0f dB')
#        plt.title('Spectrum')
#        plt.tight_layout()
#        plt.show()

        ##save feature file
        pickle.dump(log_mel_feature, open(feature_filename, 'wb' ) )
    
##print('-----DONE')    
timer.stop()
log.foot(time=timer.elapsed()) 

# In[ ]:Do zero mean and unit variance normalization for feature size (128, 431)
'''If feature size has more channel than 1 (128, 431,1)
   normalization more complecated
   This code use for 1 channel 
'''

print('-----Calculated zero mean and unit variance Normalization')  

normalization_filename_path = normalization_path + 'normalization.cpickle'

if not os.path.isfile(normalization_filename_path):  
    
    # Accummulate mean of all training set
    feature_mean_accummulate = None
    feature_std_accummulate = None
    for i, filename in enumerate(filename_list_train):
        feature_filename = feature_path + filename + '.cpickle'
        # load feature
        features = pickle.load(open(feature_filename, 'rb'))  
        # mean of each file
        feature_mean = np.mean(features,axis=1).reshape(-1,1)
        
        # Accummulate mean for all feature of training set
        if feature_mean_accummulate is None:
            feature_mean_accummulate = feature_mean
        else:
            feature_mean_accummulate +=feature_mean
            
        # sdt of each file 
        '''std = sqrt(mean(abs(x - x.mean())**2))
        '''
        feature_std = np.std(features,axis=1,ddof=0).reshape(-1,1)
#        #feature_sqrt_var = np.mean((np.abs(features-feature_mean))**2, axis=1).reshape(-1,1)
        
#        #if feature_std.all() == feature_sqrt_var.all():
#        #    print('feature_std = feature_sqrt_var')
            
        # Accummulate std for all feature of training set
        if feature_std_accummulate is None:
            feature_std_accummulate = feature_std
        else:
            feature_std_accummulate +=feature_std
        
            
    # Finalize features_mean_accummulate
    feature_mean_finalize = (feature_mean_accummulate / len(filename_list_train)).reshape(-1,1)
    # Finalize features_std_accummulate
    feature_std_finalize = (feature_std_accummulate / len(filename_list_train)).reshape(-1,1)
        
    
    stats = {'mean': feature_mean_finalize,
             'std': feature_std_finalize
            }   
    
    log.line('feature_mean_finalize shape {}'.format(feature_mean_finalize.shape))
#    log.line('feature_mean_finalize {}'.format(feature_mean_finalize))
    log.line('feature_std_finalize shape {}'.format(feature_std_finalize.shape))
#    log.line('feature_std_finalize {}'.format(feature_std_finalize))

    ##save feature file
    pickle.dump(stats, open(normalization_filename_path, 'wb' ) )
        
print('-----DONE') 


# In[ ]: Prepare data for training 


log.section_header('Training')
timer.start()               
log.line('-----Loading data for training')

stats = pickle.load(open(normalization_filename_path, 'rb')) 
#load log-mel spectrograms
LM_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list_train):    
    feature_filename = feature_path + filename + '.cpickle'
    feature = pickle.load(open(feature_filename, 'rb'))
    if normalization_flag == True:
        feature = (feature -  stats['mean'])/stats['std']
    LM_train[i,:,:,:]= feature[:,:,None]

LM_val = np.zeros((len(wavpaths_val),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
for i, filename in enumerate(filename_list_val):    
    feature_filename = feature_path + filename + '.cpickle'
    feature = pickle.load(open(feature_filename, 'rb'))
    if normalization_flag == True:
        feature = (feature -  stats['mean'])/stats['std']
    LM_val[i,:,:,:]= feature[:,:,None]

log.line('-----DONE')
log.line('Training set information')
log.line('Training set size {}'.format(LM_train.shape), indent=2)
log.line()
log.line('Validation(Test) set information')
log.line('Validation set size {}'.format(LM_val.shape), indent=2)
timer.stop()
log.foot(time=timer.elapsed()) 


# In[ ]: Normalize data for training 

        
# In[ ]: Compile model
        
model_structures = [
        model_best2019_base1(LM_train,n_labels=NumClasses, wd=1e-3),
#        model_best2019_base2(LM_train,n_labels=NumClasses, wd=1e-3)
        ]
for model_name_, model_structure_  in zip(model_names, model_structures):
    
    model_filename_path = model_path + model_name_   #'model_best2019_base2.h5'

    if not os.path.isfile(model_filename_path):        
        #create and compile the model
        model = model_structure_
        model.compile(loss='categorical_crossentropy',
                      optimizer = Adadelta(lr=max_lr),
                      metrics=['accuracy'])
        model.summary()
        
        
        # In[ ]: Train model
        
        
        #set learning rate schedule
        #lr_scheduler = LR_WarmRestart(nbatch=np.ceil(LM_train.shape[0]/batch_size), Tmult=2,
        #                              initial_lr=max_lr, min_lr=max_lr*1e-4,
        #                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]) 
        checkpoint = ModelCheckpoint(model_filename_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        ##ProgressLoggerCallback, StasherCallback=checkpoint
        callbacks = [checkpoint]
        
        #create data generator
        TrainDataGen = MixupGenerator(LM_train, 
                                      y_train, 
                                      batch_size=batch_size,
                                      alpha=mixup_alpha,
                                      )()
        
        #train the model
        history = model.fit_generator(TrainDataGen,
                                      validation_data=(LM_val, y_val),
                                      epochs=num_epochs, 
                                      verbose=1, 
                                      callbacks=callbacks,
                                      steps_per_epoch=np.ceil(LM_train.shape[0]/batch_size)
                                      ) 
        
        
        # In[ ]: Save model
        
        
        model.save(model_filename_path)
    #print('-----DONE') 
    timer.stop()
    log.foot(time=timer.elapsed()) 

# In[ ]: Inferrence
log.section_header('Testing')
timer.start()

#load filenames and labels
dev_test_df = pd.read_csv(ValFile,sep='\t', encoding='ASCII')
Inds_device_a=np.where(dev_test_df['filename'].str.contains("-a.wav")==True)[0]
Inds_device_b=np.where(dev_test_df['filename'].str.contains("-b.wav")==True)[0]
Inds_device_c=np.where(dev_test_df['filename'].str.contains("-c.wav")==True)[0]
Inds_device_bc=np.concatenate((Inds_device_b,Inds_device_c),axis=-1)

wavpaths = dev_test_df['filename'].tolist()
ClassNames = np.unique(dev_test_df['scene_label'])
y_val_labels =  dev_test_df['scene_label'].astype('category').cat.codes.values
y_val_labels_noswap23 = y_val_labels

##swap codes for 2 and 1 to match the DCASE ordering of classes
#a1=np.where(y_val_labels==2)
#a2=np.where(y_val_labels==3)
#y_val_labels.setflags(write=1)
#y_val_labels[a1] = 3
#y_val_labels[a2] = 2
#
#print(ClassNames)
#print('encoded_y_true noswap{}'.format(y_val_labels_noswap23))
#print('encoded_y_true swap23{}'.format(y_val_labels))
#print('length of test set {}'.format(len(wavpaths)))


y_val_labels = y_val_labels_noswap23 
# In[]: Prepare data for testing, (run on the training step)

#log.section_header('Testing')
#timer.start()               
#print('-----Loading data for training')
#
#stats = pickle.load(open(normalization_filename_path, 'rb')) 
#
##load log-mel spectrograms
#LM_val = np.zeros((len(wavpaths_val),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
#for i, filename in enumerate(filename_list_val):    
#    feature_filename = feature_path + filename + '.cpickle'
#    feature = pickle.load(open(feature_filename, 'rb'))
#    if normalization_flag == True:
#        feature = (feature -  stats['mean'])/stats['std']
#    LM_val[i,:,:,:]= feature[:,:,None]
#
#log.line('Validation(Test) set information')
#log.line('Validation set size {}'.format(LM_val.shape), indent=2)
#timer.stop()
#log.foot(time=timer.elapsed())


# In[5]:


# load model
# Get model filename
model_filename_path = model_path + model_names[0]
# Initialize model to None, load when first non-tested file encountered.
keras_model = None
keras_model = keras.models.load_model(model_filename_path)

# Get results filename
#result_filename_path = result_path + os.path.splitext()  #'model_best2019_base1.cpickle'


# In[6]:


#load and run the model
best_model = keras.models.load_model(model_filename_path)
y_pred_val = np.argmax(best_model.predict(LM_val),axis=1)


# In[7]:

print('-------------------------------')
#get metrics for all devices combined
Overall_accuracy = np.sum(y_pred_val==y_val_labels)/LM_val.shape[0]
print("overall accuracy all: ", Overall_accuracy)

plot_confusion_matrix(y_val_labels, y_pred_val, ClassNames,normalize=True,title="Task 1b, all devices")

conf_matrix = confusion_matrix(y_val_labels,y_pred_val)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)


# In[8]:

print('-------------------------------')
#get metrics for device A only
Overall_accuracy_A = np.sum(y_pred_val[Inds_device_a]==y_val_labels[Inds_device_a])/len(Inds_device_a)
print("overall accuracy Device A: ", Overall_accuracy_A)

plot_confusion_matrix(y_val_labels[Inds_device_a], y_pred_val[Inds_device_a], ClassNames,normalize=True,title="Task 1b, Device A")

conf_matrix = confusion_matrix(y_val_labels[Inds_device_a],y_pred_val[Inds_device_a])
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)


# In[9]:

print('-------------------------------')
#get metrics for device B only
Overall_accuracy_B = np.sum(y_pred_val[Inds_device_b]==y_val_labels[Inds_device_b])/len(Inds_device_b)
print("overall accuracy Device B: ", Overall_accuracy_B)

plot_confusion_matrix(y_val_labels[Inds_device_b], y_pred_val[Inds_device_b], ClassNames,normalize=True,title="Task 1b, Device B")

conf_matrix = confusion_matrix(y_val_labels[Inds_device_b],y_pred_val[Inds_device_b])
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)


# In[10]:

print('-------------------------------')
#get metrics for device C only
Overall_accuracy_C = np.sum(y_pred_val[Inds_device_c]==y_val_labels[Inds_device_c])/len(Inds_device_c)
print("overall accuracy Device C: ", Overall_accuracy_C)

plot_confusion_matrix(y_val_labels[Inds_device_c], y_pred_val[Inds_device_c], ClassNames,normalize=True,title="Task 1b, Device C")

conf_matrix = confusion_matrix(y_val_labels[Inds_device_c],y_pred_val[Inds_device_c])
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)


# In[11]:

print('-------------------------------')
#get metrics for device B and C 
Overall_accuracy_BC = np.sum(y_pred_val[Inds_device_bc]==y_val_labels[Inds_device_bc])/len(Inds_device_bc)
print("overall accuracy Device  B and C: ", Overall_accuracy_BC)

plot_confusion_matrix(y_val_labels[Inds_device_bc], y_pred_val[Inds_device_bc], ClassNames,normalize=True,title="Task 1b, Device B and C")

conf_matrix = confusion_matrix(y_val_labels[Inds_device_bc],y_pred_val[Inds_device_bc])
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)



#if __name__ == "__main__":
#    try:
#        sys.exit(main(sys.argv))
#
#    except (ValueError, IOError) as e:
#        sys.exit(e)