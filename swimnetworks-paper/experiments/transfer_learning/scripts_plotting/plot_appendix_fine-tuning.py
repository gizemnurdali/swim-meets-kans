"""
This script generates the figure in the appendix that compares the performance
of the sampling and Adam training approaches before and after fine-tuning for
different widths for different pre-trained models

Output: 
-------
fine_tuning.png
"""

import csv
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################
########################### 1) ResNet50 #####################################
##########################################################################
data_folder_name = '../data_paper'

# Re-training
with open(os.path.join(os.path.dirname(__file__),data_folder_name , 'ResNet50_training_ft.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

model = 'resnet'
weights = 'retrain_'
exec(weights + model + '_train_0' "=data[1]") #, 0:7
exec(weights + model + "_test_0 = data[2]")
exec(weights + model + "_train_ft = data[4]")
exec(weights + model + "_test_ft = data[5]")

# Sampling
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'ResNet50_sampling_ft.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)
model = 'resnet'
weights = 'sample_'

exec(weights + model + '_train_0' "=data[1]") #, 0:7
exec(weights + model + "_test_0 = data[2]")
exec(weights + model + "_train_ft = data[4]")
exec(weights + model + "_test_ft = data[5]")


##########################################################################
########################### 2) VGG19 #####################################
##########################################################################

# Re-training
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'VGG19_training_ft.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

model = 'vgg19'
weights = 'retrain_'
exec(weights + model + '_train_0' "=data[1]") #, 0:7
exec(weights + model + "_test_0 = data[2]")
exec(weights + model + "_train_ft = data[4]")
exec(weights + model + "_test_ft = data[5]")

# Sampling
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'VGG19_sampling_ft.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)
model = 'vgg19'
weights = 'sample_'
exec(weights + model + '_train_0' "=data[1]") #, 0:7
exec(weights + model + "_test_0 = data[2]")
exec(weights + model + "_train_ft = data[4]")
exec(weights + model + "_test_ft = data[5]")

##########################################################################
########################### 3) Xception #####################################
##########################################################################

# Re-training
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'Xception_training_ft.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)

model = 'xception'
weights = 'retrain_'
exec(weights + model + '_train_0' "=data[1]") #, 0:7
exec(weights + model + "_test_0 = data[2]")
exec(weights + model + "_train_ft = data[4]")
exec(weights + model + "_test_ft = data[5]")

# Sampling
with open(os.path.join(os.path.dirname(__file__),data_folder_name, 'Xception_sampling_ft.csv'), 'r') as file:
    reader = csv.reader(file)
    data = np.array(list(reader)).astype(float)
model = 'xception'
weights = 'sample_'
exec(weights + model + '_train_0' "=data[1]") #, 0:7
exec(weights + model + "_test_0 = data[2]")
exec(weights + model + "_train_ft = data[4]")
exec(weights + model + "_test_ft = data[5]")

# Generate the figure with fine-tuning the plots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(ncols=3, nrows=2, figsize=(10, 8), sharex=True, sharey=True,)
w = 4
x = [1, 2, 3]
labels = ['ResNet50', 'VGG19', 'Xception']
widths = [512, 1024, 2048, 4096, 6144, 8192]

# Plot the data on the subplots: Error bars
for i in range(len(widths)):
        i += 1
        sampling = np.array([sample_resnet_test_0[i-1] * 100., 
                  sample_vgg19_test_0[i-1] * 100., 
                  sample_xception_test_0[i-1] * 100.])
        sampling_ft = np.array([sample_resnet_test_ft[i-1] * 100. - sample_resnet_test_0[i-1] * 100., 
                  sample_vgg19_test_ft[i-1] * 100. - sample_vgg19_test_0[i-1] * 100., 
                  sample_xception_test_ft[i-1] * 100. - sample_xception_test_0[i-1] * 100.])
        adam_tr = np.array([retrain_resnet_test_0[i-1] * 100., 
                  retrain_vgg19_test_0[i-1] * 100., 
                  retrain_xception_test_0[i-1] * 100.])
        adam_tr_ft = np.array([retrain_resnet_test_ft[i-1] * 100. - retrain_resnet_test_0[i-1] * 100., 
                  retrain_vgg19_test_ft[i-1] * 100. - retrain_vgg19_test_0[i-1] * 100., 
                  retrain_xception_test_ft[i-1] * 100. - retrain_xception_test_0[i-1] * 100.])
        
        with sns.axes_style("white"):
            sns.set_style("ticks")
            sns.set_context("talk")
            
            # plot details
            bar_width = 0.35
            epsilon = .015
            line_width = 1
            opacity = 0.7
            pos_bar_positions = np.arange(len(adam_tr))
            neg_bar_positions = pos_bar_positions + bar_width
            exec('ax' + str(i) + '.bar(pos_bar_positions, adam_tr, bar_width, capsize=6, color="#E37222", edgecolor="#E37222",linewidth=line_width)')#,label="Adam"
            exec('ax' + str(i) + '.bar(pos_bar_positions, adam_tr_ft, bar_width, capsize=6,bottom=adam_tr, alpha=opacity, color="white", edgecolor="#E37222", linewidth=line_width, hatch="//")')#, label="Adam + Fine-tuning"
            exec('ax' + str(i) + '.bar(neg_bar_positions, sampling, bar_width, capsize=6, color="#005293", edgecolor="#005293", linewidth=line_width)')#, label="Sampling"
            exec('ax' + str(i) + '.bar(neg_bar_positions, sampling_ft , bar_width, capsize=6, bottom=sampling, color="white", hatch="//", edgecolor="#005293", linewidth=line_width)')#, label="Sampling + Fine-tuning"

            exec('ax' + str(i) + ".set_ylim([80, 100])")
            exec('ax' + str(i) + ".set_xticks((neg_bar_positions + pos_bar_positions)/2, labels, rotation=45)")
            exec('ax' + str(i) + ".set_title('Width: " + str(widths[i-1]) +  "', fontsize=10)")

legend_1 = ["Adam", "Adam + Fine-tuning", "Sampling", "Sampling + Fine-tuning"] #"Sampling + Fine-tuning"
exec('ax' + str(1) + ".set_ylabel('Test accuracy')")
exec('ax' + str(4) + ".set_ylabel('Test accuracy')")
exec('fig' + ".legend(labels= legend_1, ncols = 4, bbox_to_anchor=(0.5, 0.94), loc='center', fontsize=10, borderaxespad=0.1,)")

plt.savefig(os.path.join(os.path.dirname(__file__), 'fine_tuning.png'))
plt.close()

