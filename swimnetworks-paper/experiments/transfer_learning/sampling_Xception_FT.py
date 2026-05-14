'''
This script deals with two phases of transfer learning:
- (1) Feature Extraction: In the first phase, the sampling algorithm is used for finding weigths 
of the hidden layer of the classification head. 
- (2) Fine-tuning: In the second phase, the entire network (including the pre-trained layers) 
are trained iteratively with the Adam optimizer

Target task: ImageNet

Source task: CIFAR-10

Pre-trained neural network architecture: Xception

Weights and biases of the classification head: Sampling algorithm

Fine-tuning: Adam optimizer with a low learning rate of 1e-5

Outputs: 'Xception_sampling_ft.csv' file with 
- Feature extraction phase: accuracy on training data, test data and sampling time for different widths in the feature extraction phase
- Fine-tuning: accuracy on training data, test data
'''

import numpy as np
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sys
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from swimnetworks import Dense, Linear
from utils import swim_to_keras_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from keras import Sequential
from keras.applications import Xception
from keras.preprocessing.image import ImageDataGenerator
from time import time

#Lastly import the final layers that will be added on top of the base model'
from keras.layers import GlobalAveragePooling2D

# Import to_categorical from the keras utils package to one hot encode the labels'
from keras.utils import to_categorical

# Uncomment the following line to choose a particular GPU from the available ones
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
##################################################################
############### Data Management ##################################
##################################################################
# Load Data
from keras.datasets import cifar10

# Split Data in Train, Validation and Test Datasets
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train = x_train.astype('float32')[0:1000]
x_test = x_test.astype('float32')[0:1000] 
y_train = y_train[0:1000]
y_test = y_test[0:1000]

# Preprocessing data
x_train = tf.keras.applications.xception.preprocess_input(x_train)
x_test = tf.keras.applications.xception.preprocess_input(x_test)

# Upsampling makes sense and is helpful (ImageNet has larger images)
x_train = tf.image.resize(x_train, (224, 224), method='bicubic')
x_test = tf.image.resize(x_test, (224, 224), method='bicubic')

# Print the shapes of train, validation, test and split datasets
print('Training data: ', (x_train.shape, y_train.shape))
print('Test data: ', (x_test.shape, y_test.shape))

##################################################################
############### Data Augmentation ############################
##################################################################
train_generator = ImageDataGenerator(
        rotation_range=10,  
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
base_model_retraining = Xception(include_top=False,weights='imagenet',input_shape=(224, 224,3),classes=y_train.shape[1])
base_model_retraining.trainable = True
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
widths = [512, 1024, 2048, 4096, 6144, 8192]
activation_functions = [1] # 1:tanh, 2:relu
train_acc_ = []
test_acc_ = []
sampling_time_ = []
train_acc_finetuning_ = []
test_acc_finetuning_ = []
# Define early stopping criterion
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True, # Stop at the lowest value of validation loss
    start_from_epoch=0,
)

# Sample the head and compute accuracy for different activation functions and widths
for a in range(len(activation_functions)):
    for i in range(len(widths)):
        # Select the activation function and parameter sampler
        if a == 0:
            activation_function = "tanh"
            param_sampler = "tanh"
        else:
            activation_function = "relu"
            param_sampler = "relu"
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

        # Get a tensorflow model
        print(np.shape(i_train)[1])
        model_tf = swim_to_keras_model(sampled_head, input_shape=(np.shape(i_train)[1]), set_weights=True)#loss='mse', optimizer=tf.optimizers.Adam(learning_rate=1e-5)

        # Evaluate accuracy on training dataset
        pred_train = sampled_head.transform(i_train)
        train_acc = accuracy_score(y_train, pred_train)

        # Evaluate accuracy on test dataset
        pred_test = sampled_head.transform(i_test)
        test_acc = accuracy_score(y_test, pred_test)

        # Fine-tuning the entire model 
        base_model_retraining = Xception(include_top=False,weights='imagenet',input_shape=(224, 224,3),classes=y_train.shape[1])
        base_model_retraining.trainable = True
        model_finetune= Sequential()
        model_finetune.add(base_model_retraining) 
        model_finetune.add(GlobalAveragePooling2D()) 
        model_finetune.add(model_tf)
        model_finetune.compile(loss='mse', optimizer=tf.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])
        model_finetune.summary()

        epochs = 10
        history_fintetuning = model_finetune.fit(x_train, to_categorical(y_train), 
                              validation_data=(x_test, to_categorical(y_test)), 
                              epochs=epochs, 
                              callbacks = [early_stopping],
                              verbose=1) # Default batch size = 32
        
        # Evaluate accuracy on training dataset
        pred_train_ft = np.argmax(model_finetune.predict(x_train), axis=1)
        train_acc_finetuning = accuracy_score(y_train, pred_train_ft)

        # Evaluate accuracy on test dataset
        pred_test_ft = np.argmax(model_finetune.predict(x_test), axis=1)
        test_acc_finetuning = accuracy_score(y_test, pred_test_ft)
        
        # Update all the lists
        train_acc_.append(train_acc)
        test_acc_.append(test_acc)
        sampling_time_.append(sampling_time)
        train_acc_finetuning_.append(train_acc_finetuning)
        test_acc_finetuning_.append(test_acc_finetuning)
        
        # Summary
        print('Summary: Pre-trained neural network - Xception, Dataset: CIFAR-10, Head Weights - Sampled')
        print('Width: ', widths[i])
        print('Activation function: ', activation_function)
        print('Training accuracy before finetuning: ', train_acc)
        print('Test accuracy before finetuning: ', test_acc)
        print('Training accuracy after finetuning: ', train_acc_finetuning)
        print('Test accuracy after finetuning: ', test_acc_finetuning)
        print('Sampling time: ', sampling_time, '\n')
        
        # Delete the current model
        keras.backend.clear_session()
        del model_finetune


data = [train_acc_, test_acc_, sampling_time_, train_acc_finetuning_, test_acc_finetuning_]
data = pd.DataFrame(data)
data.to_csv('Xception_sampling_ft.csv', index=False)

