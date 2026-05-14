"""
This script generates the figure in the appendix which compares the train and test 
accuracies obtained with 1 and 2 hidden layers in the classification head for 
different widths.

Description of Figure 5:
-------------------------
Left: Sampling 
Right: Adam training

Output: 
-------
layers.png
"""

import csv
import sys, os
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
########################### 1) Read data #################################
##########################################################################
data_folder_name = '../data_paper'

with open(os.path.join(os.path.dirname(__file__),data_folder_name , 'ResNet50_sampling_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

widths = [64, 512, 1024, 2048, 4096, 6144, 8192]
n_widths = len(widths)
exec('sampling_train_1' + "= data[1, 0:n_widths]")
exec('sampling_test_1' + "= data[2, 0:n_widths]")

with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'ResNet50_sampling_2layers.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

exec('sampling_train_2' + "= data[1, 0:n_widths]")
exec('sampling_test_2' + "= data[2, 0:n_widths]")
    
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'ResNet50_training_5.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

widths = [64, 512, 1024, 2048, 4096, 6144, 8192]
n_widths = len(widths)
exec('retraining_train_1' + "= data[1, 0:n_widths]")
exec('retraining_test_1' + "= data[2, 0:n_widths]")

with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'ResNet50_training_2layers.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

exec('retraining_train_2' + "= data[1, 0:n_widths]")
exec('retraining_test_2' +  "= data[2, 0:n_widths]")

##########################################################################
########################### 2) Create Plot ###############################
##########################################################################
# Plot the data
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8.5, 5), sharey=True)

# Plot the data on the subplots
ax1.plot(widths, sampling_train_1 * 100.0, 
         color='#0065BD', marker='^',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax1.plot(widths, sampling_test_1 * 100.0, 
         color='#0065BD', marker='D',
         linewidth=2, markersize=6, alpha = 1.0)
ax1.plot(widths, sampling_train_2 * 100.0,
         color='red', marker='o',linestyle='--',
         linewidth=2, markersize=6,  alpha = 0.8)
ax1.plot(widths, sampling_test_2 * 100.0,
         color='red', marker='X',
         linewidth=2, markersize=6,  alpha = 0.8)

ax1.set_xscale('log')
ax1.set_ylim([80, 100])
ax1.set_xlabel('Width')
ax1.set_ylabel('Accuracy')

# Plot the data on the subplots
ax2.plot(widths, retraining_train_1 * 100.0,
         color='#0065BD', marker='^',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, retraining_test_1 * 100.0, 
         color='#0065BD', marker='D',
         linewidth=2, markersize=6, alpha = 0.8)
ax2.plot(widths, retraining_train_2 * 100.0,
         color='red', marker='o',linestyle='--',
         linewidth=2, markersize=6,  alpha = 0.8)
ax2.plot(widths, retraining_test_2 * 100.0,
         color='red', marker='X',
         linewidth=2, markersize=6, alpha = 0.8)

ax2.set_xscale('log')
ax2.set_ylim([80, 100])
ax2.set_xlabel('Width')

legend_1 = ["1-Layer: Train data", "1-Layer: Test data", "2-Layers: Train data", "2-Layers: Test data"] #"Sampling + Fine-tuning"
exec('fig' + ".legend(labels= legend_1, ncols = 4, bbox_to_anchor=(0.5, 0.95), loc='center', fontsize=10, borderaxespad=0.1,)")
plt.savefig(os.path.join(os.path.dirname(__file__), 'layers.png'))
plt.close()
