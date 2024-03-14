import os
import sys

import logic
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

results_path = os.path.dirname(dataset_path) + "_Processed/" + os.path.basename(dataset_path)
if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(f"{results_path}/color"):
    os.makedirs(f"{results_path}/color")
if not os.path.exists(f"{results_path}/grayscale"):
    os.makedirs(f"{results_path}/grayscale")
if not os.path.exists(f"{results_path}/combined"):
    os.makedirs(f"{results_path}/combined")
if not os.path.exists(f"{results_path}/train"):
    os.makedirs(f"{results_path}/train")
if not os.path.exists(f"{results_path}/test"):
    os.makedirs(f"{results_path}/test")
if not os.path.exists(f"{results_path}/val"):
    os.makedirs(f"{results_path}/val")

##############################
# MAIN LOGIC                 #
##############################
# Prepare the inputs.
dataset = inputs.load_dataset(dataset_path)

models = logic.define_models(img_shape)
train, test, val = logic.preprocess(dataset, dataset_split, img_shape)

# Train & test the GAN model.
logic.train(results_path, models, train, epochs, train_samples)
logic.test(inputs.get_best_model(results_path), test_samples, img_shape)

# Outputs of the models.
outputs.plot_outputs()