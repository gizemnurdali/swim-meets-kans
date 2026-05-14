"""
This script generates the Figure 5 in the main paper 

Description of Figure 5:
-------------------------
Left: Train and test accuracies with different widths for ResNet50 
(averaged over 5 random seeds). 

Middle: Test accuracy with different models with and without fine-tuning 
(width = 2048).

Right: Adam training and sampling times of the classification head 
(averaged over 5 experiments).

Output: 
-------
comparison_errorbar.png
"""
import csv
import sys, os
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
##### 1.1) Adam and Adam + Finetuning results: Read Data #####
##########################################################################

# Re-training
# Read the 5 files and store the 5 values from the files
model = 'resnet'
weights = 'retrain_'
width = 2048
data_folder_name = '../data_paper'
num_seeds = 5
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name ,'ResNet50_training_ft_' + str(i+1) + '_' + str(width) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + '_train_0' "=data[1]") #, 0:7
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_0 = data[2]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_train_ft = data[4]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_ft = data[5]")


model = 'vgg'
weights = 'retrain_'
width = 2048
num_seeds = 5
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'VGG19_training_ft_' + str(i+1) + '_' + str(width) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + '_train_0' "=data[1]") #, 0:7
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_0 = data[2]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_train_ft = data[4]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_ft = data[5]")


model = 'xception'
weights = 'retrain_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'Xception_training_ft_' + str(i+1) + '_' + str(width) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + '_train_0' "=data[1]") #, 0:7
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_0 = data[2]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_train_ft = data[4]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_ft = data[5]")

#######################################################################################
##### 1.2) Adam and Adam + Finetuning results: Compute avg, min, max from 5 seeds #####
#######################################################################################
resnet_adam_2048_avg = np.mean(np.array((retrain_resnet_2048_1_test_0,
                               retrain_resnet_2048_2_test_0,
                               retrain_resnet_2048_3_test_0,
                               retrain_resnet_2048_4_test_0,
                               retrain_resnet_2048_5_test_0)))
     
resnet_adam_2048_min = np.amin(np.array((retrain_resnet_2048_1_test_0,
                               retrain_resnet_2048_2_test_0,
                               retrain_resnet_2048_3_test_0,
                               retrain_resnet_2048_4_test_0,
                               retrain_resnet_2048_5_test_0)))

resnet_adam_2048_max = np.amax(np.array((retrain_resnet_2048_1_test_0,
                               retrain_resnet_2048_2_test_0,
                               retrain_resnet_2048_3_test_0,
                               retrain_resnet_2048_4_test_0,
                               retrain_resnet_2048_5_test_0)))

resnet_adam_FT_2048_avg = np.mean(np.array((retrain_resnet_2048_1_test_ft,
                               retrain_resnet_2048_2_test_ft,
                               retrain_resnet_2048_3_test_ft,
                               retrain_resnet_2048_4_test_ft,
                               retrain_resnet_2048_5_test_ft)))

resnet_adam_FT_2048_min = np.amin(np.array((retrain_resnet_2048_1_test_ft,
                               retrain_resnet_2048_2_test_ft,
                               retrain_resnet_2048_3_test_ft,
                               retrain_resnet_2048_4_test_ft,
                               retrain_resnet_2048_5_test_ft)))

resnet_adam_FT_2048_max = np.amax(np.array((retrain_resnet_2048_1_test_ft,
                               retrain_resnet_2048_2_test_ft,
                               retrain_resnet_2048_3_test_ft,
                               retrain_resnet_2048_4_test_ft,
                               retrain_resnet_2048_5_test_ft)))


vgg_adam_2048_avg = np.mean(np.array((retrain_vgg_2048_1_test_0,
                               retrain_vgg_2048_2_test_0,
                               retrain_vgg_2048_3_test_0,
                               retrain_vgg_2048_4_test_0,
                               retrain_vgg_2048_5_test_0)))

vgg_adam_2048_min = np.amin(np.array((retrain_vgg_2048_1_test_0,
                               retrain_vgg_2048_2_test_0,
                               retrain_vgg_2048_3_test_0,
                               retrain_vgg_2048_4_test_0,
                               retrain_vgg_2048_5_test_0)))

vgg_adam_2048_max = np.amax(np.array((retrain_vgg_2048_1_test_0,
                               retrain_vgg_2048_2_test_0,
                               retrain_vgg_2048_3_test_0,
                               retrain_vgg_2048_4_test_0,
                               retrain_vgg_2048_5_test_0)))
    

vgg_adam_FT_2048_avg = np.mean(np.array((retrain_vgg_2048_1_test_ft,
                               retrain_vgg_2048_2_test_ft,
                               retrain_vgg_2048_3_test_ft,
                               retrain_vgg_2048_4_test_ft,
                               retrain_vgg_2048_5_test_ft)))

vgg_adam_FT_2048_min = np.amin(np.array((retrain_vgg_2048_1_test_ft,
                               retrain_vgg_2048_2_test_ft,
                               retrain_vgg_2048_3_test_ft,
                               retrain_vgg_2048_4_test_ft,
                               retrain_vgg_2048_5_test_ft)))
vgg_adam_FT_2048_max = np.amax(np.array((retrain_vgg_2048_1_test_ft,
                               retrain_vgg_2048_2_test_ft,
                               retrain_vgg_2048_3_test_ft,
                               retrain_vgg_2048_4_test_ft,
                               retrain_vgg_2048_5_test_ft)))

xception_adam_2048_avg = np.mean(np.array((retrain_xception_2048_1_test_0,
                               retrain_xception_2048_2_test_0,
                               retrain_xception_2048_3_test_0,
                               retrain_xception_2048_4_test_0,
                               retrain_xception_2048_5_test_0)))

xception_adam_2048_min = np.amin(np.array((retrain_xception_2048_1_test_0,
                               retrain_xception_2048_2_test_0,
                               retrain_xception_2048_3_test_0,
                               retrain_xception_2048_4_test_0,
                               retrain_xception_2048_5_test_0)))

xception_adam_2048_max = np.amax(np.array((retrain_xception_2048_1_test_0,
                               retrain_xception_2048_2_test_0,
                               retrain_xception_2048_3_test_0,
                               retrain_xception_2048_4_test_0,
                               retrain_xception_2048_5_test_0)))   

xception_adam_FT_2048_avg = np.mean(np.array((retrain_xception_2048_1_test_ft,
                               retrain_xception_2048_2_test_ft,
                               retrain_xception_2048_3_test_ft,
                               retrain_xception_2048_4_test_ft,
                               retrain_xception_2048_5_test_ft)))

xception_adam_FT_2048_min = np.amin(np.array((retrain_xception_2048_1_test_ft,
                               retrain_xception_2048_2_test_ft,
                               retrain_xception_2048_3_test_ft,
                               retrain_xception_2048_4_test_ft,
                               retrain_xception_2048_5_test_ft)))

xception_adam_FT_2048_max = np.amax(np.array((retrain_xception_2048_1_test_ft,
                               retrain_xception_2048_2_test_ft,
                               retrain_xception_2048_3_test_ft,
                               retrain_xception_2048_4_test_ft,
                               retrain_xception_2048_5_test_ft)))                       

#########################################################################
##### 2.1) Sampling and + Finetuning results: Read Data #####
##########################################################################

# Sampling
# Read the 5 files and store the 5 values from the files
model = 'resnet'
weights = 'sample_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'ResNet50_sampling_ft_' + str(i+1) + '_' + str(width) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + '_train_0' "=data[1]") #, 0:7
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_0 = data[2]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_train_ft = data[4]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_ft = data[5]")


model = 'vgg'
weights = 'sample_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'VGG19_sampling_ft_' + str(i+1) + '_' + str(width) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + '_train_0' "=data[1]") #, 0:7
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_0 = data[2]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_train_ft = data[4]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_ft = data[5]")


model = 'xception'
weights = 'sample_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'Xception_sampling_ft_' + str(i+1) + '_' + str(width) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + '_train_0' "=data[1]") #, 0:7
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_0 = data[2]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_train_ft = data[4]")
    exec(weights + model + '_' + str(width) + '_' + str(i + 1) + "_test_ft = data[5]")

#############################################################################################
##### 2.2) Sampling and + Finetuning results: Compute avg, min, max from 5 random seeds #####
#############################################################################################
resnet_sample_2048_avg = np.mean(np.array((sample_resnet_2048_1_test_0,
                               sample_resnet_2048_2_test_0,
                               sample_resnet_2048_3_test_0,
                               sample_resnet_2048_4_test_0,
                               sample_resnet_2048_5_test_0)))
     
resnet_sample_2048_min = np.amin(np.array((sample_resnet_2048_1_test_0,
                               sample_resnet_2048_2_test_0,
                               sample_resnet_2048_3_test_0,
                               sample_resnet_2048_4_test_0,
                               sample_resnet_2048_5_test_0)))

resnet_sample_2048_max = np.amax(np.array((sample_resnet_2048_1_test_0,
                               sample_resnet_2048_2_test_0,
                               sample_resnet_2048_3_test_0,
                               sample_resnet_2048_4_test_0,
                               sample_resnet_2048_5_test_0)))

resnet_sample_FT_2048_avg = np.mean(np.array((sample_resnet_2048_1_test_ft,
                               sample_resnet_2048_2_test_ft,
                               sample_resnet_2048_3_test_ft,
                               sample_resnet_2048_4_test_ft,
                               sample_resnet_2048_5_test_ft)))

resnet_sample_FT_2048_min = np.amin(np.array((sample_resnet_2048_1_test_ft,
                               sample_resnet_2048_2_test_ft,
                               sample_resnet_2048_3_test_ft,
                               sample_resnet_2048_4_test_ft,
                               sample_resnet_2048_5_test_ft)))

resnet_sample_FT_2048_max = np.amax(np.array((sample_resnet_2048_1_test_ft,
                               sample_resnet_2048_2_test_ft,
                               sample_resnet_2048_3_test_ft,
                               sample_resnet_2048_4_test_ft,
                               sample_resnet_2048_5_test_ft)))


vgg_sample_2048_avg = np.mean(np.array((sample_vgg_2048_1_test_0,
                               sample_vgg_2048_2_test_0,
                               sample_vgg_2048_3_test_0,
                               sample_vgg_2048_4_test_0,
                               sample_vgg_2048_5_test_0)))
    
vgg_sample_2048_min = np.amin(np.array((sample_vgg_2048_1_test_0,
                               sample_vgg_2048_2_test_0,
                               sample_vgg_2048_3_test_0,
                               sample_vgg_2048_4_test_0,
                               sample_vgg_2048_5_test_0)))

vgg_sample_2048_max = np.amax(np.array((sample_vgg_2048_1_test_0,
                               sample_vgg_2048_2_test_0,
                               sample_vgg_2048_3_test_0,
                               sample_vgg_2048_4_test_0,
                               sample_vgg_2048_5_test_0))) 

vgg_sample_FT_2048_avg = np.mean(np.array((sample_vgg_2048_1_test_ft,
                               sample_vgg_2048_2_test_ft,
                               sample_vgg_2048_3_test_ft,
                               sample_vgg_2048_4_test_ft,
                               sample_vgg_2048_5_test_ft)))
vgg_sample_FT_2048_min = np.amin(np.array((sample_vgg_2048_1_test_ft,
                               sample_vgg_2048_2_test_ft,
                               sample_vgg_2048_3_test_ft,
                               sample_vgg_2048_4_test_ft,
                               sample_vgg_2048_5_test_ft)))
vgg_sample_FT_2048_max = np.amax(np.array((sample_vgg_2048_1_test_ft,
                               sample_vgg_2048_2_test_ft,
                               sample_vgg_2048_3_test_ft,
                               sample_vgg_2048_4_test_ft,
                               sample_vgg_2048_5_test_ft)))

xception_sample_2048_avg = np.mean(np.array((
                               sample_xception_2048_1_test_0,
                               sample_xception_2048_2_test_0,
                               sample_xception_2048_3_test_0,
                               sample_xception_2048_4_test_0,
                               sample_xception_2048_5_test_0)))
     

xception_sample_2048_min = np.amin(np.array((
                               sample_xception_2048_1_test_0,
                               sample_xception_2048_2_test_0,
                               sample_xception_2048_3_test_0,
                               sample_xception_2048_4_test_0,
                               sample_xception_2048_5_test_0)))

xception_sample_2048_max = np.amax(np.array((
                               sample_xception_2048_1_test_0,
                               sample_xception_2048_2_test_0,
                               sample_xception_2048_3_test_0,
                               sample_xception_2048_4_test_0,
                               sample_xception_2048_5_test_0)))


xception_sample_FT_2048_avg = np.mean(np.array((
                               sample_xception_2048_1_test_ft,
                               sample_xception_2048_2_test_ft,
                               sample_xception_2048_3_test_ft,                               sample_xception_2048_2_test_ft,
                               sample_xception_2048_4_test_ft,
                               sample_xception_2048_5_test_ft)))


xception_sample_FT_2048_min = np.amin(np.array((
                               sample_xception_2048_1_test_ft,
                               sample_xception_2048_2_test_ft,
                               sample_xception_2048_3_test_ft,
                               sample_xception_2048_4_test_ft,
                               sample_xception_2048_5_test_ft)))

xception_sample_FT_2048_max = np.amax(np.array((
                               sample_xception_2048_1_test_ft,
                               sample_xception_2048_2_test_ft,
                               sample_xception_2048_3_test_ft,
                               sample_xception_2048_4_test_ft,
                               sample_xception_2048_5_test_ft)))


##############################################################
###### 3.1) Plot on the left and right: Read data ###### 
##############################################################

model = 'resnet'
weights = 'retrain_'
width = 2048
num_seeds = 5

for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'ResNet50_training_' + str(i+1) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(i + 1) + '_train' "=data[1,0:7]") #, 0:7
    exec(weights + model + '_' + str(i + 1) + "_test = data[2,0:7]")
    exec(weights + model + '_' + str(i + 1) + "_time = data[3,0:7]")

model = 'resnet'
weights = 'sample_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'ResNet50_sampling_' + str(i+1) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(i + 1) + '_train' "=data[1,0:7]") #, 0:7
    exec(weights + model + '_' + str(i + 1) + "_test = data[2,0:7]")
    exec(weights + model + '_' + str(i + 1) + "_time = data[3,0:7]")


model = 'vgg'
weights = 'retrain_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'VGG19_training_' + str(i+1) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(i + 1) + '_train' "=data[1,0:7]") #, 0:7
    exec(weights + model + '_' + str(i + 1) + "_test = data[2,0:7]")
    exec(weights + model + '_' + str(i + 1) + "_time = data[3,0:7]")

model = 'vgg'
weights = 'sample_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'VGG19_sampling_' + str(i+1) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(i + 1) + '_train' "=data[1,0:7]") #, 0:7
    exec(weights + model + '_' + str(i + 1) + "_test = data[2,0:7]")
    exec(weights + model + '_' + str(i + 1) + "_time = data[3,0:7]")

model = 'xception'
weights = 'retrain_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'Xception_training_' + str(i+1) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(i + 1) + '_train' "=data[1,0:7]") #, 0:7
    exec(weights + model + '_' + str(i + 1) + "_test = data[2,0:7]")
    exec(weights + model + '_' + str(i + 1) + "_time = data[3,0:7]")

model = 'xception'
weights = 'sample_'
width = 2048
for i in range(num_seeds):
    with open(os.path.join(os.path.dirname(__file__), data_folder_name, 'Xception_sampling_' + str(i+1) + '.csv'), 'r') as file:
        reader = csv.reader(file)
        data = np.array(list(reader)).astype(float)
    exec(weights + model + '_' + str(i + 1) + '_train' "=data[1,0:7]") #, 0:7
    exec(weights + model + '_' + str(i + 1) + "_test = data[2,0:7]")
    exec(weights + model + '_' + str(i + 1) + "_time = data[3,0:7]")

########################################################################
###### 3.2) Plot on the left and right: Compute mean from 5 seeds ###### 
########################################################################

retrain_vgg_train_avg = np.mean(np.array((retrain_vgg_1_train,
                               retrain_vgg_2_train,
                               retrain_vgg_3_train,
                               retrain_vgg_4_train,
                               retrain_vgg_5_train)), axis=0)

retrain_vgg_test_avg = np.mean(np.array((retrain_vgg_1_test,
                               retrain_vgg_2_test,
                               retrain_vgg_3_test,
                               retrain_vgg_4_test,
                               retrain_vgg_5_test)), axis=0)

retrain_vgg_time_avg = np.mean(np.array((retrain_vgg_1_time,
                               retrain_vgg_2_time,
                               retrain_vgg_3_time,
                               retrain_vgg_4_time,
                               retrain_vgg_5_time)), axis=0)
 

retrain_xception_train_avg = np.mean(np.array((retrain_xception_1_train,
                               retrain_xception_2_train,
                               retrain_xception_3_train,
                               retrain_xception_4_train,
                               retrain_xception_5_train)), axis=0)

retrain_xception_test_avg = np.mean(np.array((retrain_xception_1_test,
                               retrain_xception_2_test,
                               retrain_xception_3_test,
                               retrain_xception_4_test,
                               retrain_xception_5_test)), axis=0)

retrain_xception_time_avg = np.mean(np.array((retrain_xception_1_time,
                               retrain_xception_2_time,
                               retrain_xception_3_time,
                               retrain_xception_4_time,
                               retrain_xception_5_time)), axis=0)


retrain_resnet_train_avg = np.mean(np.array((retrain_resnet_1_train,
                               retrain_resnet_2_train,
                               retrain_resnet_3_train,
                               retrain_resnet_4_train,
                               retrain_resnet_5_train)), axis=0)

retrain_resnet_test_avg = np.mean(np.array((retrain_resnet_1_test,
                               retrain_resnet_2_test,
                               retrain_resnet_3_test,
                               retrain_resnet_4_test,
                               retrain_resnet_5_test)), axis=0)

retrain_resnet_time_avg = np.mean(np.array((retrain_resnet_1_time,
                               retrain_resnet_2_time,
                               retrain_resnet_3_time,
                               retrain_resnet_4_time,
                               retrain_resnet_5_time)), axis=0)

sample_resnet_train_avg = np.mean(np.array((sample_resnet_1_train,
                               sample_resnet_2_train,
                               sample_resnet_3_train,
                               sample_resnet_4_train,
                               sample_resnet_5_train)), axis=0)

sample_resnet_test_avg = np.mean(np.array((sample_resnet_1_test,
                               sample_resnet_2_test,
                               sample_resnet_3_test,
                               sample_resnet_4_test,
                               sample_resnet_5_test)), axis=0)

sample_resnet_time_avg = np.mean(np.array((sample_resnet_1_time,
                               sample_resnet_2_time,
                               sample_resnet_3_time,
                               sample_resnet_4_time,
                               sample_resnet_5_time)), axis=0)


sample_vgg_train_avg = np.mean(np.array((sample_vgg_1_train,
                               sample_vgg_2_train,
                               sample_vgg_3_train,
                               sample_vgg_4_train,
                               sample_vgg_5_train)), axis=0)

sample_vgg_test_avg = np.mean(np.array((sample_vgg_1_test,
                               sample_vgg_2_test,
                               sample_vgg_3_test,
                               sample_vgg_4_test,
                               sample_vgg_5_test)), axis=0)

sample_vgg_time_avg = np.mean(np.array((sample_vgg_1_time,
                               sample_vgg_2_time,
                               sample_vgg_3_time,
                               sample_vgg_4_time,
                               sample_vgg_5_time)), axis=0)

sample_xception_train_avg = np.mean(np.array((sample_xception_1_train,
                               sample_xception_2_train,
                               sample_xception_3_train,
                               sample_xception_4_train,
                               sample_xception_5_train)), axis=0)

sample_xception_test_avg = np.mean(np.array((sample_xception_1_test,
                               sample_xception_2_test,
                               sample_xception_3_test,
                               sample_xception_4_test,
                               sample_xception_5_test)), axis=0)

sample_xception_time_avg = np.mean(np.array((sample_resnet_1_time,
                               sample_xception_2_time,
                               sample_xception_3_time,
                               sample_xception_4_time,
                               sample_xception_5_time)), axis=0)

############################
###### (4) Main Plot  ######
############################

# Plot the data
# Create a figure with two subplots
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))
# Plot the data on the subplots

# 1: Train and test accuracy for different widths for the Adam training approach and sampling approach for ResNet50
widths = [64, 512, 1024, 2048, 4096, 6144, 8192]

# Plot the data on the subplots
ax1.plot(widths, retrain_resnet_test_avg * 100.0, label='Adam: Test data',
         color='#E37222', marker='^',
         linewidth=2, markersize=6, alpha = 0.8)
ax1.plot(widths, retrain_resnet_train_avg * 100.0, label='Adam: Train data',
         color='#E37222', marker='D',
         linewidth=2, markersize=6, linestyle='--', alpha = 0.8)
ax1.plot(widths, sample_resnet_test_avg * 100.0, label='Sampling: Test data', 
         color='#0065BD', marker='o',
         linewidth=2, markersize=6, alpha = 0.8)
ax1.plot(widths, sample_resnet_train_avg * 100.0, label='Sampling: Train data', 
         color='#0065BD', marker='X',linestyle='--',
         linewidth=2, markersize=6, alpha = 0.8)

ax1.set_xscale('log')
ax1.set_ylim([80, 100])
ax1.set_xlabel('Width')
ax1.set_ylabel('Accuracy')
ax1.legend(bbox_to_anchor=(0.01, 1.3), loc='upper left', fontsize=10)


# 2 : Test accuracy with different models for the Adam training approach and sampling approach with and without fine-tuning.
marker_size = 30
x = [1, 2, 3]
labels = ['ResNet50', 'VGG19', 'Xception']

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

retrained_values_0 = [resnet_adam_2048_avg * 100., 
                  vgg_adam_2048_avg * 100., 
                  xception_adam_2048_avg * 100.]
retrained_values_ft = [resnet_adam_FT_2048_avg * 100., 
                  vgg_adam_FT_2048_avg * 100., 
                  xception_adam_FT_2048_avg * 100.]
sampled_values_0 = [resnet_sample_2048_avg * 100., 
                  vgg_sample_2048_avg * 100., 
                  xception_sample_2048_avg * 100.]
sampled_values_ft = [resnet_sample_FT_2048_avg * 100., 
                  vgg_sample_FT_2048_avg * 100., 
                  xception_sample_FT_2048_avg * 100.]

adam_tr = np.array([resnet_adam_2048_avg * 100., 
                  vgg_adam_2048_avg * 100., 
                  xception_adam_2048_avg * 100.])
adam_tr_ft = np.array([ resnet_adam_FT_2048_avg * 100.- resnet_adam_2048_avg * 100., 
                   vgg_adam_FT_2048_avg * 100.- vgg_adam_2048_avg * 100., 
                  xception_adam_FT_2048_avg * 100.- xception_adam_2048_avg * 100.])
sampling = np.array([resnet_sample_2048_avg * 100., 
                  vgg_sample_2048_avg * 100., 
                  xception_sample_2048_avg * 100.])
sampling_ft = np.array([resnet_sample_FT_2048_avg * 100. - resnet_sample_2048_avg * 100., 
                  vgg_sample_FT_2048_avg * 100. - vgg_sample_2048_avg * 100., 
                  xception_sample_FT_2048_avg * 100. - xception_sample_2048_avg * 100.])

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

    adam_min = adam_tr - np.array((resnet_adam_2048_min, 
                                     vgg_adam_2048_min, 
                                     xception_adam_2048_min)) * 100.
  
    adam_max = np.array((resnet_adam_2048_max, 
                                     vgg_adam_2048_max, 
                                     xception_adam_2048_max,)) * 100. - adam_tr
       
    adam_ft_min = adam_tr + adam_tr_ft - np.array((resnet_adam_FT_2048_min, 
                                     vgg_adam_FT_2048_min, 
                                     xception_adam_FT_2048_min,)) * 100.
                
    adam_ft_max = np.array((resnet_adam_FT_2048_max,
                            vgg_adam_FT_2048_max, 
                            xception_adam_FT_2048_max)) * 100. - (adam_tr + adam_tr_ft)
    
        
    sampling_min = sampling - np.array((resnet_sample_2048_min, 
                                     vgg_sample_2048_min, 
                                     xception_sample_2048_min,)) * 100.
  
    sampling_max = np.array((resnet_sample_2048_max, 
                        vgg_sample_2048_max, 
                        xception_sample_2048_max,)) * 100. - sampling
    
    sampling_ft_min = sampling + sampling_ft - np.array((resnet_sample_FT_2048_min, 
                                     vgg_sample_FT_2048_min, 
                                     xception_sample_FT_2048_min,)) * 100.
                
    sampling_ft_max = np.array((resnet_sample_FT_2048_max,
                            vgg_sample_FT_2048_max, 
                            xception_sample_FT_2048_max)) * 100. - (sampling + sampling_ft)
    
      
    adam_err = [adam_min, adam_max]
    adam_ft_err = [adam_ft_min, adam_ft_max]
    sampling_err = [sampling_min, sampling_max]
    sampling_ft_err = [sampling_ft_min, sampling_ft_max]
    
    # make bar plots
    adam_bar = ax2.bar(pos_bar_positions, adam_tr, bar_width,
                       yerr = adam_err,capsize=6,
                              color='#E37222',
                              edgecolor='#E37222',
                              linewidth=line_width,
                              label='Adam')
    adam_ft_bar = ax2.bar(pos_bar_positions, adam_tr_ft, bar_width,
                              yerr = adam_ft_err,capsize=6,
                              bottom=adam_tr,
                              alpha=opacity,
                              color='white',
                              edgecolor='#E37222',
                              linewidth=line_width,
                              hatch='//',
                              label='Adam + Fine-tuning')
    sampling_bar = ax2.bar(neg_bar_positions, sampling, bar_width,
                           yerr = sampling_err,capsize=6,
                              color='#005293', #''#ED0020',#
                              edgecolor='#005293',
                              linewidth=line_width,
                              label='Sampling')
    sampling_ft_bar = ax2.bar(neg_bar_positions, sampling_ft , bar_width,
                              yerr = sampling_ft_err,capsize=6,
                              bottom=sampling,
                              color="white",
                              hatch='//',
                              edgecolor='#005293',
                              #ecolor='#ED0020',
                              linewidth=line_width,
                              label='Sampling + Fine-tuning')
    ax2.set_xticks((neg_bar_positions + pos_bar_positions)/2, labels, rotation=45)
    ax2.set_ylim([80, 100])

    ax2.set_ylabel('Test accuracy')
    ax2.legend(bbox_to_anchor=(0.01, 1.3), loc='upper left', fontsize=10)
    sns.despine()
    fig.tight_layout()



x = [1, 2, 3]
labels = ['ResNet50', 'VGG19', 'Xception']

# Plot the data on the subplots
ax3.plot(widths, retrain_resnet_time_avg, label='Adam: ResNet50',
         color='#E37222',
         linewidth=2, markersize=6, linestyle='solid', alpha = 0.8)
ax3.plot(widths, retrain_vgg_time_avg, label='Adam: VGG19',
         color='#E37222',
         linewidth=2, markersize=6, linestyle='dashed', alpha = 0.8)
ax3.plot(widths, retrain_xception_time_avg, label='Adam: Xception',
         color='#E37222',
         linewidth=2, markersize=6, linestyle='dotted', alpha = 0.8)
ax3.plot(widths, sample_resnet_time_avg, label='Sampling: ResNet50',
         color='#0065BD', linestyle='solid',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, sample_vgg_time_avg, label='Sampling: VGG19',
         color='#0065BD', linestyle='dashed',
         linewidth=2, markersize=6, alpha = 0.8)
ax3.plot(widths, sample_xception_time_avg, label='Sampling: Xception',
         color='#0065BD', linestyle='dotted',
         linewidth=2, markersize=6, alpha = 0.8)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Width')
ax3.set_ylabel('Time')
ax3.legend(bbox_to_anchor=(-0.25, 1.3), loc='upper left', fontsize=10, ncols = 2)

fig.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'comparison_errorbar.png'))
plt.close()
