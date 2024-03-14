import os
import sys

import test
import train
import inputs
import outputs
import preprocess

if sys.argc <= 1:
    print("Please provide path to a dataset.")
    exit()

dataset_path = sys.argv[1]
if not os.path.exists(dataset_path):
    print("The dataset path does not exists.")
    exit()

##############################
# PARAMETERS                 #
##############################
channels      = 3
img_rows      = 256
img_cols      = 256
val_percent   = 0.2
test_percent  = 0.2
train_percent = 0.6

# Compile the parameters into compact variables.
img_shape     = (img_rows, img_cols, channels)
dataset_split = (train_percent, test_percent, val_percent)
results_path  = os.path.dirname(dataset_path) + "_Processed/" + os.path.basename(dataset_path)

##############################
# PRE-PROCESS                #
# Process the database.      #
##############################
datasets = preprocess(dataset_path, results_path, dataset_split, img_shape)
train_dataset, test_dataset, val_dataset = datasets

##############################
# INPUTS                     #
# Prepare the inputs.        #
##############################
models  = define_models(img_shape)
dataset = inputs.load_batch(f"{path}/train", train_samples)

##############################
# TRAIN & TEST               #
# Train the GAN model.       #
##############################
train(results_path, models, dataset, epochs, train_samples)
test(get_best_model(results_path), test_samples, img_shape)

##############################
# OUTPUTS                    #
# Outputs of the models.     #
##############################
outputs.plot_outputs()