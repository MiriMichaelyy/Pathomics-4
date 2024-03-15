#!/bin/python3
import os
import sys

import logic
import Models
import inputs
import outputs

if len(sys.argv) <= 1:
    print("Missing dataset argument.")
    exit()

dataset_path = os.path.abspath(sys.argv[1])
if not os.path.exists(dataset_path):
    print(f"The dataset path ({dataset_path}) does not exists.")
    exit()

##############################
# PARAMETERS                 #
##############################
preprocess    = True
channels      = 3
img_rows      = 256
img_cols      = 256
val_percent   = 0.0
test_percent  = 0.3
train_percent = 0.7
epochs        = 15

# Compile the parameters into compact variables.
img_shape     = (img_rows, img_cols, channels)
dataset_split = (train_percent, test_percent, val_percent)

# Create directories.
processed_path = os.path.join(dataset_path, "Processed")
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
if not os.path.exists(os.path.join(processed_path, "color")):
    os.makedirs(os.path.join(processed_path, "color"))
if not os.path.exists(os.path.join(processed_path, "grayscale")):
    os.makedirs(os.path.join(processed_path, "grayscale"))
if not os.path.exists(os.path.join(processed_path, "combined")):
    os.makedirs(os.path.join(processed_path, "combined"))
if not os.path.exists(os.path.join(processed_path, "train", "color")):
    os.makedirs(os.path.join(processed_path, "train", "color"))
if not os.path.exists(os.path.join(processed_path, "train", "grayscale")):
    os.makedirs(os.path.join(processed_path, "train", "grayscale"))
if not os.path.exists(os.path.join(processed_path, "test", "color")):
    os.makedirs(os.path.join(processed_path, "test", "color"))
if not os.path.exists(os.path.join(processed_path, "test", "grayscale")):
    os.makedirs(os.path.join(processed_path, "test", "grayscale"))
if not os.path.exists(os.path.join(processed_path, "val", "color")):
    os.makedirs(os.path.join(processed_path, "val", "color"))
if not os.path.exists(os.path.join(processed_path, "val", "grayscale")):
    os.makedirs(os.path.join(processed_path, "val", "grayscale"))

results_path = os.path.join(dataset_path, "Results")
if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(os.path.join(results_path, "models")):
    os.makedirs(os.path.join(results_path, "models"))

##############################
# PREPROCESS DATASETS        #
##############################
# Generate basic processed dataset.
total = 0
print(f"Splitting original images into {img_cols}x{img_rows} (color, grayscale & combined).")
for image in inputs.load_dataset(dataset_path):
    color, grayscale, combined = logic.process_image(image, img_shape)
    total += outputs.save_originals(processed_path, color, grayscale, combined, total)

# Load dataset generators.
print("Loading color & grayscale datasets.")
color     = inputs.load_dataset(os.path.join(processed_path, "color"))
grayscale = inputs.load_dataset(os.path.join(processed_path, "grayscale"))

# Create sup-datasets.
print(f"Splitting datasets into train ({train_percent}%), test ({test_percent}%) and val ({val_percent}%).")
datasets         = (color, grayscale)
dataset_size     = len(os.listdir(os.path.join(processed_path, "color")))
train, test, val = logic.split_dataset(datasets, dataset_size, dataset_split)

# Save processed datasets.
outputs.save_split(processed_path, train, test, val)

##############################
# MAIN LOGIC                 #
##############################
# Train & test the GAN model.
print("Starting to train models.")
color     = inputs.load_dataset(os.path.join(results_path, "train", "color"))
grayscale = inputs.load_dataset(os.path.join(results_path, "train", "grayscale"))

models = Models.define_models(img_shape)
for epoch in range(epochs):

    # Split the dataset into equal batches.
    # Convert the image into numpy arrays.
    batch_size = dataset_size // epochs
    color_arr, gray_arr = inputs.load_batch(zip(color, grayscale), batch_size)

    # Train the model and calculate losses.
    print(f"Epoch #{epoch+1} | Batch: {epoch * batch_size} - {(epoch + 1) * batch_size}")
    models, losses = logic.train(models, color_arr, gray_arr)

    # Save the losses and models in the results directory.
    outputs.save_losses(losses, results_path, epoch + 1)
    outputs.save_models(models, results_path, epoch + 1)

print("Starting best model test.")
# test  = inputs.load_dataset(os.path.join(results_path, "test"))
# model = inputs.get_best_model(results_path)
# logic.test(model, test)

# Outputs of the models.
outputs.plot_outputs(*losses)