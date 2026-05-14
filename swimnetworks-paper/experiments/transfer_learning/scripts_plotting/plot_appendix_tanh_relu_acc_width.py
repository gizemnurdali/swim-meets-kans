"""
This script generates three figures:

(1) acc_width.png: compares the train and test accuracy for different widths (number of neurons in the hidden
layer of the classification head) using three pre-trained neural network architectures for the Adam557
training and sampling approaches

(2) tanh_relu_sampling.png: Compares the performance of the sampling algorithm used to sample the weights of
the hidden layer of the classification head for tanh and ReLu activation function

(3) tanh_relu_training.png: Compares the performance of the Adam training algorithm used to find weights of
the hidden layer of the classification head for tanh and ReLu activation function

Output: 
-------
acc_width.png
tanh_relu_sampling.png
tanh_relu_training.png
"""

import csv
import sys, os
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
########################### 1) ResNet50 #####################################
##########################################################################
data_folder_name = '../data_paper'

# Re-training
with open(os.path.join(os.path.dirname(__file__),data_folder_name , 'ResNet50_training_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

model = 'resnet'
weights = 'retrain_'
exec(weights + model + '_train_tanh' "=data[1, 0:7]")
exec(weights + model + "_train_relu = data[1, 7:14]")
exec(weights + model + "_test_tanh = data[2, 0:7]")
exec(weights + model + "_test_relu = data[2, 7:14]")
exec(weights + model + "_time_tanh = data[3, 0:7]")
exec(weights + model + "_time_relu = data[3, 7:14]")

# Sampling
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'ResNet50_sampling_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)
model = 'resnet'
weights = 'sample_'
exec(weights + model + '_train_tanh' + "= data[1, 0:7]")
exec(weights + model + "_train_relu = data[1, 7:14]")
exec(weights + model + "_test_tanh = data[2, 0:7]")
exec(weights + model + "_test_relu = data[2, 7:14]")
exec(weights + model + "_time_tanh = data[3, 0:7]")
exec(weights + model + "_time_relu = data[3, 7:14]")


# Accuracy Vs Width plots for Xception
plt.figure()
# Plot the data on the subplots
widths = [64, 512, 1024, 2048, 4096, 6144, 8192]
plt.plot(widths, retrain_resnet_train_tanh * 100.0, label='retraining: train data, tanh', 
         color='red', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)
plt.plot(widths, retrain_resnet_test_tanh * 100.0, label='retraining: test data, tanh', 
         color='blue', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)
plt.plot(widths, retrain_resnet_train_relu * 100.0, label='retraining: train data, relu', 
         color='green', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)
plt.plot(widths, retrain_resnet_test_relu * 100.0, label='retraining: test data, relu', 
         color='orange', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)

plt.legend()
plt.xscale('log')
plt.ylim([80, 100])
plt.xlabel('width')
plt.ylabel('accuracy')
plt.savefig(os.path.join(os.path.dirname(__file__),'tanh_vs_relu_resnet50_trained.png'))
plt.close()


##########################################################################
########################### 2) VGG19 #####################################
##########################################################################

# Re-training
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'VGG19_training_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

model = 'vgg19'
weights = 'retrain_'
exec(weights + model + '_train_tanh' "=data[1, 0:7]")
exec(weights + model + "_train_relu = data[1, 7:14]")
exec(weights + model + "_test_tanh = data[2, 0:7]")
exec(weights + model + "_test_relu = data[2, 7:14]")
exec(weights + model + "_time_tanh = data[3, 0:7]")
exec(weights + model + "_time_relu = data[3, 7:14]")

# Sampling
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'VGG19_sampling_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)
model = 'vgg19'
weights = 'sample_'
exec(weights + model + '_train_tanh' + "= data[1, 0:7]")
exec(weights + model + "_train_relu = data[1, 7:14]")
exec(weights + model + "_test_tanh = data[2, 0:7]")
exec(weights + model + "_test_relu = data[2, 7:14]")
exec(weights + model + "_time_tanh = data[3, 0:7]")
exec(weights + model + "_time_relu = data[3, 7:14]")

##########################################################################
########################### 3) Xception #####################################
##########################################################################

# Re-training
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'Xception_training_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

model = 'xception'
weights = 'retrain_'
exec(weights + model + '_train_tanh' "=data[1, 0:7]")
exec(weights + model + "_train_relu = data[1, 7:14]")
exec(weights + model + "_test_tanh = data[2, 0:7]")
exec(weights + model + "_test_relu = data[2, 7:14]")
exec(weights + model + "_time_tanh = data[3, 0:7]")
exec(weights + model + "_time_relu = data[3, 7:14]")

# Sampling
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'Xception_sampling_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)
model = 'xception'
weights = 'sample_'
exec(weights + model + '_train_tanh' + "= data[1, 0:7]")
exec(weights + model + "_train_relu = data[1, 7:14]")
exec(weights + model + "_test_tanh = data[2, 0:7]")
exec(weights + model + "_test_relu = data[2, 7:14]")
exec(weights + model + "_time_tanh = data[3, 0:7]")
exec(weights + model + "_time_relu = data[3, 7:14]")


##########################################################################
######## Accuracy Vs Width (Combined plot): ResNet50, VGG19, Xception ######
##########################################################################

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5), sharey=True)
ax1.plot(widths, retrain_resnet_test_tanh * 100.0,
         color='#E37222', marker='^',
         linewidth=2, markersize=6, alpha = 0.8)#label='Adam: test data',
ax1.plot(widths, retrain_resnet_train_tanh * 100.0, 
         color='#E37222', marker='D',
         linewidth=2, markersize=6, linestyle='--', alpha = 0.8)#label='Adam: train data',
ax1.plot(widths, sample_resnet_test_tanh * 100.0,  
         color='#0065BD', marker='o',
         linewidth=2, markersize=6, alpha = 0.8)#label='Sampling: test data',
ax1.plot(widths, sample_resnet_train_tanh * 100.0,  
         color='#0065BD', marker='X',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)#label='Sampling: train data',

ax1.set_xscale('log')
ax1.set_ylim([80, 100])
ax1.set_xlabel('Width')
ax1.set_ylabel('Accuracy')

#ax1.legend(bbox_to_anchor=(-0.05, 1.0), loc='upper right')
#ax1.set_title('ResNet50: accuracy Vs width')


ax2.plot(widths, retrain_vgg19_test_tanh * 100.0, 
         color='#E37222', marker='^',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, retrain_vgg19_train_tanh * 100.0, 
         color='#E37222', marker='D',
         linewidth=2, markersize=6, linestyle='--', alpha = 0.8)
ax2.plot(widths, sample_vgg19_test_tanh * 100.0, 
         color='#0065BD', marker='o',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, sample_vgg19_train_tanh * 100.0, 
         color='#0065BD', marker='X',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)

ax2.set_xscale('log')
ax2.set_ylim([80, 100])
ax2.set_xlabel('Width')
#ax2.legend(bbox_to_anchor=(-0.05, 1.0), loc='upper right')
#ax2.set_title('ResNet50: accuracy Vs width')



ax3.plot(widths, retrain_xception_test_tanh * 100.0,
         color='#E37222', marker='^',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, retrain_xception_train_tanh * 100.0, 
         color='#E37222', marker='D',
         linewidth=2, markersize=6, linestyle='--', alpha = 0.8)
ax3.plot(widths, sample_xception_test_tanh * 100.0,  
         color='#0065BD', marker='o',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, sample_xception_train_tanh * 100.0, 
         color='#0065BD', marker='X',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)

ax3.set_xscale('log')
ax3.set_ylim([80, 100])
ax3.set_xlabel('Width')
#ax3.legend(bbox_to_anchor=(-0.05, 1.0), loc='upper right')
#ax3.set_title('ResNet50: accuracy Vs width')
legend_1 = ["Adam: Test data", "Adam: Train data", "Sampling: Test data", "Sampling: Train data"] #"Sampling + Fine-tuning"
exec('fig' + ".legend(labels= legend_1, ncols = 4, bbox_to_anchor=(0.5, 0.95), loc='center', fontsize=10, borderaxespad=0.1,)")
#fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'acc_width.png'))
plt.close()

##########################################################################
######## tanh Vs ReLU Sampling: ResNet50, VGG19, Xception ####################
##########################################################################
# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(10, 5), sharey=True)
# Plot the data on the subplots
ax1.plot(widths, sample_resnet_train_tanh * 100.0, 
         color='#0065BD', marker='^',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax1.plot(widths, sample_resnet_test_tanh * 100.0,
         color='#0065BD', marker='D',
         linewidth=2, markersize=6, alpha = 0.8)
ax1.plot(widths, sample_resnet_train_relu * 100.0,
         color='green', marker='o',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax1.plot(widths, sample_resnet_test_relu * 100.0,
         color='green', marker='X',
         linewidth=2, markersize=6, alpha = 0.8)

ax1.set_xscale('log')
ax1.set_ylim([80, 100])
ax1.set_xlabel('Width')
ax1.set_ylabel('Accuracy')
#ax1.set_title('ResNet50')

#ax1.legend()

ax2.plot(widths, sample_vgg19_train_tanh * 100.0,
         color='#0065BD', marker='^',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, sample_vgg19_test_tanh * 100.0, 
         color='#0065BD', marker='D',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, sample_vgg19_train_relu * 100.0,
         color='green', marker='o',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, sample_vgg19_test_relu * 100.0,
         color='green', marker='X',
         linewidth=2, markersize=6, alpha = 0.8)

ax2.set_xscale('log')
ax2.set_ylim([80, 100])
ax2.set_xlabel('Width')
#ax2.set_ylabel('accuracy')
#ax2.set_title('VGG19')

#ax2.legend()


ax3.plot(widths, sample_xception_train_tanh * 100.0, 
         color='#0065BD', marker='^',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, sample_xception_test_tanh * 100.0, 
         color='#0065BD', marker='D',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, sample_xception_train_relu * 100.0,
         color='green', marker='o',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, sample_xception_test_relu * 100.0, 
         color='green', marker='X',
         linewidth=2, markersize=6, alpha = 0.8)

ax3.set_xscale('log')
ax3.set_ylim([80, 100])
ax3.set_xlabel('Width')
#ax3.set_ylabel('accuracy')
#ax3.set_title('Xception')

#ax3.legend()
legend_1 = ["tanh: Train data", "tanh: Test data", "ReLU: Train data", "ReLU: Test data"] #"Sampling + Fine-tuning"
exec('fig' + ".legend(labels= legend_1, ncols = 4, bbox_to_anchor=(0.5, 0.95), loc='center', fontsize=10, borderaxespad=0.1,)")

# Add a legend
#plt.legend()
#fig.suptitle('Transfer learning with sampling: tanh Vs ReLU', fontsize=12)
#fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'tanh_relu_sampling.png'))
plt.close()


##########################################################################
######## tanh Vs ReLU, retrained : ResNet50, VGG19, Xception ####################
##########################################################################
# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
# Plot the data on the subplots
ax1.plot(widths, retrain_resnet_train_tanh * 100.0, label='tanh: train data', 
         color='red', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)
ax1.plot(widths, retrain_resnet_test_tanh * 100.0, label='tanh: test data', 
         color='orange', marker='X',
         linewidth=2, markersize=8, alpha = 0.8)
ax1.plot(widths, retrain_resnet_train_relu * 100.0, label='ReLU: train data',
         color='purple', marker='+',
         linewidth=2, markersize=8, alpha = 0.8)
ax1.plot(widths, retrain_resnet_test_relu * 100.0, label='ReLU: test data',
         color='brown', marker='D',
         linewidth=2, markersize=8, alpha = 0.8)

ax1.set_xscale('log')
ax1.set_ylim([80, 100])
ax1.set_xlabel('width')
ax1.set_ylabel('accuracy')
#ax1.set_title('ResNet50')

ax1.legend()

ax2.plot(widths, retrain_vgg19_train_tanh * 100.0, label='tanh: train data', 
         color='red', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)
ax2.plot(widths, retrain_vgg19_test_tanh * 100.0, label='tanh: test data', 
         color='orange', marker='X',
         linewidth=2, markersize=8, alpha = 0.8)
ax2.plot(widths, retrain_vgg19_train_relu * 100.0, label='ReLU: train data',
         color='purple', marker='+',
         linewidth=2, markersize=8, alpha = 0.8)
ax2.plot(widths, retrain_vgg19_test_relu * 100.0, label='ReLU: test data',
         color='brown', marker='D',
         linewidth=2, markersize=8, alpha = 0.8)

ax2.set_xscale('log')
ax2.set_ylim([80, 100])
ax2.set_xlabel('width')
ax2.set_ylabel('accuracy')
#ax2.set_title('VGG19')

ax2.legend()


ax3.plot(widths, retrain_xception_train_tanh * 100.0, label='tanh: train data', 
         color='red', marker='o',
         linewidth=2, markersize=8, alpha = 0.8)
ax3.plot(widths, retrain_xception_test_tanh * 100.0, label='tanh: test data', 
         color='orange', marker='X',
         linewidth=2, markersize=8, alpha = 0.8)
ax3.plot(widths, retrain_xception_train_relu * 100.0, label='ReLU: train data',
         color='purple', marker='+',
         linewidth=2, markersize=8, alpha = 0.8)
ax3.plot(widths, retrain_xception_test_relu * 100.0, label='ReLU: test data',
         color='brown', marker='D',
         linewidth=2, markersize=8, alpha = 0.8)

ax3.set_xscale('log')
ax3.set_ylim([80, 100])
ax3.set_xlabel('width')
ax3.set_ylabel('accuracy')
#ax3.set_title('Xception')

ax3.legend()

# Add a legend
#plt.legend()
#fig.suptitle('Transfer learning with sampling: tanh Vs ReLU', fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'tanh_relu_training.png'))
plt.close()