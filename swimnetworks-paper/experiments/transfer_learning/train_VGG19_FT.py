'''
This script deals with two phases of transfer learning:
- (1) Feature Extraction: In the first phase, the hidden layers of the classification head
are trained using the Adam optimizer
- (2) Fine-tuning: In the second phase, the entire network (including the pre-trained layers) 
are trained iteratively with the Adam optimizer

Target task: ImageNet

Source task: CIFAR-10

Pre-trained neural network architecture: VGG19

Weights and biases of the classification head: Sampling algorithm

Fine-tuning: Adam optimizer with a low learning rate of 1e-5

Outputs: 'ResNet50_sampling_ft.csv' file with 
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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from time import time

#Lastly import the final layers that will be added on top of the base model'
from keras.layers import Dense, GlobalAveragePooling2D

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
(x_train,y_train), (x_test,y_test)=cifar10.load_data()
x_train= x_train.astype('float32')
x_test = x_test.astype('float32') 

# Need to convert this to categorical for training (In sampling, it's done inside the layers)
y_train_original = y_train.copy() 
y_test_original = y_test.copy() 
y_train=to_categorical(y_train) 
y_test=to_categorical(y_test) 

# Super Important: Preprocessing data - Inputs pixel values are scaled between -1 and 1, sample-wise
x_train_1 = tf.keras.applications.vgg19.preprocess_input(x_train)
x_test_1 = tf.keras.applications.vgg19.preprocess_input(x_test)

# Upsampling makes sense and is helpful (ImageNet has larger images)
x_train = tf.image.resize(x_train_1, (224, 224), method='bicubic')
x_test = tf.image.resize(x_test_1, (224, 224), method='bicubic')

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
base_model_retraining = VGG19(include_top=False,weights='imagenet',input_shape=(224, 224, 3),classes=y_train.shape[1])

# Freeze the weights of the pre-trained model
base_model_retraining.trainable = False

# Pass data through the pre-trained network
model_pretrained=Sequential()
model_pretrained.add(base_model_retraining) 
model_pretrained.add(GlobalAveragePooling2D()) 

# Output of the pre-trained model/feature extractor (Without the head)
i_train = model_pretrained.predict(x_train)
i_test = model_pretrained.predict(x_test)

# Hyper-parameters
widths = [2048] #512, 1024, 2048, 4096, 6144, 8192
activation_functions = [1] # 1:tanh, 2:relu
train_acc_ = []
test_acc_ = []
training_time_ = []
train_acc_finetuning_ = []
test_acc_finetuning_ = []

# batch size and epochs
batch_size = 32
epochs = 10

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
            activation_function = 'tanh'
        else:
            activation_function = 'relu'
        print ('Activation function: ', activation_function)
        
        # Head of the neural network (Fully connected layer + classifier)
        model=Sequential()
        model.add(Dense(widths[i], activation=(activation_function)))
        model.add(Dense(10, activation=('softmax'))) 
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Sample weights
        time_start = time()
        history = model.fit(i_train, y_train, 
                              validation_data=(i_test, y_test), 
                              epochs=epochs, 
                              callbacks = [early_stopping],
                              verbose=1)
        time_end = time()
        training_time = time_end - time_start

        # Evaluate accuracy on training dataset
        pred_train = np.argmax(model.predict(i_train, batch_size=batch_size), axis=1)
        train_acc = accuracy_score(y_train_original, pred_train)

        # Evaluate accuracy on test dataset
        pred_test = np.argmax(model.predict(i_test), axis=1)
        test_acc = accuracy_score(y_test_original, pred_test)

        # Fine-tuning the entire model (with a smaller learning rate)
        base_model_retraining = VGG19(include_top=False,weights='imagenet',input_shape=(224, 224, 3),classes=y_train.shape[1])
        base_model_retraining.trainable = True
        model_finetune= Sequential()
        model_finetune.add(base_model_retraining) 
        model_finetune.add(GlobalAveragePooling2D()) 
        model_finetune.add(model)
        model_finetune.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])
        model_finetune.summary()

        history_fintetuning = model_finetune.fit(x_train, y_train, 
                              validation_data=(x_test, y_test), 
                              epochs=epochs, 
                              callbacks = [early_stopping],
                              verbose=1) # Default batch size = 32
        # Evaluate accuracy on training dataset
        pred_train_ft = np.argmax(model_finetune.predict(x_train), axis=1)
        train_acc_finetuning = accuracy_score(y_train_original, pred_train_ft)

        # Evaluate accuracy on test dataset
        pred_test_ft = np.argmax(model_finetune.predict(x_test), axis=1)
        test_acc_finetuning = accuracy_score(y_test_original, pred_test_ft)
        
        # Update all the lists
        train_acc_.append(train_acc)
        test_acc_.append(test_acc)
        training_time_.append(training_time)
        train_acc_finetuning_.append(train_acc_finetuning)
        test_acc_finetuning_.append(test_acc_finetuning)
               
        # Summary
        print('Summary: Pre-trained neural network - VGG19, Dataset: CIFAR-10, Head Weights - Trained')
        print('Width: ', widths[i])
        print('Activation function: ', activation_function)
        print('Training accuracy: ', train_acc)
        print('Test accuracy: ', test_acc)
        print('Training accuracy after finetuning: ', train_acc_finetuning)
        print('Test accuracy after finetuning: ', test_acc_finetuning)
        print('Training time: ', training_time, '\n')
        
        # Delete the current model
        keras.backend.clear_session()
        del model
        del model_finetune

data = [train_acc_, test_acc_, training_time_, train_acc_finetuning_, test_acc_finetuning_]
data = pd.DataFrame(data)
data.to_csv('VGG19_training_ft.csv', index=False)
