'''
This script deals with the transfer learning task using the sampling algorithm 
for pre-trained VGG19. 

Target task: ImageNet

Source task: CIFAR-10

Pre-trained neural network architecture: VGG19 

Weights and biases of the classification head: Sampling algorithm

Outputs: 'VGG19_sampling.csv' file with accuracy on training data, test data and sampling time for different widths
'''

import numpy as np
from time import time
import tensorflow as tf
import pandas as pd
import PIL
import sys
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from swimnetworks import Dense, Linear

import numpy as np
import pandas as pd
from keras import Sequential
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from time import time
from keras.layers import GlobalAveragePooling2D 

# Uncomment the following line to choose a particular GPU from the available ones
# os.environ["CUDA_VISIBLE_DEVICES"]="2" # second gpu

##################################################################
############### Data Management ##################################
##################################################################
# Load Data
from keras.datasets import cifar10

# Split Data in Train, Validation and Test Datasets
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 

# Preprocessing data
x_train = tf.keras.applications.vgg19.preprocess_input(x_train)
x_test = tf.keras.applications.vgg19.preprocess_input(x_test)

# Upsampling makes sense and is helpful (ImageNet has larger images)
x_train = tf.image.resize(x_train, (224, 224), method='bicubic')
x_test = tf.image.resize(x_test, (224, 224), method='bicubic')

# Print the shapes of train, validation, test and split datasets
print('Training data: ', (x_train.shape, y_train.shape))
print('Test data: ', (x_test.shape, y_test.shape))

##################################################################
############### Data Augmentation ############################
##################################################################
train_generator = ImageDataGenerator(rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.1,  
        height_shift_range=0.1,
        shear_range = 0.1,
        horizontal_flip=True,  
        vertical_flip=False)
train_generator.fit(x_train)

##################################################################
############### Build Model ##################################
##################################################################

# Pre-trained model (without the head = fully-connected-layer + classifier)  
base_model_retraining = VGG19(include_top=False,weights='imagenet',input_shape=(224, 224, 3),classes=y_train.shape[1])
model_1= Sequential()
model_1.add(base_model_retraining) 
model_1.add(GlobalAveragePooling2D()) 
model_1.summary()

# Output of the pre-trained model/feature extractor (Without the head)
i_train = model_1.predict(x_train)
i_test = model_1.predict(x_test)
print('\nTraining data: Output shape after the pre-trained model: ', np.shape(i_train))
print('\nTest data: Output shape after the pre-trained model: ', np.shape(i_test))

# Hyper-parameters
widths = [64, 512, 1024, 2048, 4096, 6144, 8192]
activation_functions = [1, 2] # 1:tanh, 2:relu
train_acc_ = []
test_acc_ = []
sampling_time_ = []

# Sample the head and compute accuracy for different activation functions and widths
for a in range(len(activation_functions)):
    for i in range(len(widths)):
        # Select the activation function and parameter sampler
        if a == 0:
            activation_function = "tanh"
            param_sampler = "tanh"
        else:
            activation_function = "relu"
            param_sampler = parameter_sampler="relu"
        print ('Activation function: ', activation_function)
        
        # Head of the neural network (1 hidden layer + 1 output layer)
        steps = [
                ("fcn1", 
                Dense(layer_width=widths[i], is_classifier=True,
                            activation=activation_function, 
                            parameter_sampler=param_sampler, 
                            random_seed=5)),
                ("lin", Linear(layer_width=y_train.shape[1], is_classifier=True,
                            regularization_scale=0.0))
            ]

        sampled_head = Pipeline(steps=steps, )

        # Sample weights
        time_start = time()
        history_snn = sampled_head.fit(i_train, y_train)
        time_end = time()
        sampling_time = time_end - time_start
   
        # Evaluate accuracy on training dataset
        pred_train = sampled_head.transform(i_train)
        train_acc = accuracy_score(y_train, pred_train)

        # Evaluate accuracy on test dataset
        pred_test = sampled_head.transform(i_test)
        test_acc = accuracy_score(y_test, pred_test)

        # Update all the lists
        train_acc_.append(train_acc)
        test_acc_.append(test_acc)
        sampling_time_.append(sampling_time)
        
        # Summary
        print('Summary: Pre-trained neural network - VGG19, Dataset: CIFAR-10, Head Weights - Sampled')
        print('Width: ', widths[i])
        print('Activation function: ', activation_function)
        print('Training accuracy: ', train_acc)
        print('Test accuracy: ', test_acc)
        print('Sampling time: ', sampling_time, '\n')


data = [train_acc_, test_acc_, sampling_time_]
data = pd.DataFrame(data)
data.to_csv('VGG19_sampling.csv', index=False)

