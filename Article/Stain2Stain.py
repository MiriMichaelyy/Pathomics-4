#!/bin/python3
import os
import sys

import preprocess
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
preprocessed  = True
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
os.makedirs(processed_path,                                     exist_ok=True)
os.makedirs(os.path.join(processed_path, "color"),              exist_ok=True)
os.makedirs(os.path.join(processed_path, "grayscale"),          exist_ok=True)
os.makedirs(os.path.join(processed_path, "combined"),           exist_ok=True)
os.makedirs(os.path.join(processed_path, "train", "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "train", "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",  "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",  "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",  "generated"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",   "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",   "grayscale"), exist_ok=True)

results_path = os.path.join(dataset_path, "Results")
os.makedirs(results_path,                         exist_ok=True)
os.makedirs(os.path.join(results_path, "models"), exist_ok=True)

##############################
# PREPROCESS DATASETS        #
##############################
if not preprocessed:
    # Generate basic processed dataset.
    total = 0
    print(f"Splitting original images into {img_cols}x{img_rows} (color, grayscale & combined).")
    for image in inputs.load_dataset(dataset_path):
        color, grayscale, combined = preprocess.process_image(image, img_shape)
        total += outputs.save_originals(processed_path, color, grayscale, combined, total)

    # Load dataset generators.
    print("Loading color & grayscale datasets.")
    color     = inputs.load_dataset(os.path.join(processed_path, "color"))
    grayscale = inputs.load_dataset(os.path.join(processed_path, "grayscale"))

    # Create sup-datasets.
    print(f"Splitting datasets into train ({train_percent}%), test ({test_percent}%) and val ({val_percent}%).")
    datasets         = (color, grayscale)
    dataset_size     = inputs.get_size(os.path.join(processed_path, "color"))
    train, test, val = preprocess.split_dataset(datasets, dataset_size, dataset_split)

    # Save processed datasets.
    outputs.save_split(processed_path, train, test, val)

##############################
# MAIN LOGIC                 #
##############################
# Train & test the GAN model.
print("Starting to train models.")
color         = inputs.load_dataset(os.path.join(processed_path, "train", "color"))
grayscale     = inputs.load_dataset(os.path.join(processed_path, "train", "grayscale"))
dataset_size  = inputs.get_size(os.path.join(processed_path, "train", "color"))
dataset_size += inputs.get_size(os.path.join(processed_path, "test",  "color"))
dataset_size += inputs.get_size(os.path.join(processed_path, "val",   "color"))

models = Models.define_models(img_shape)
for epoch in range(epochs):

    # Split the dataset into equal batches.
    # Convert the image into numpy arrays.
    batch_size = dataset_size // epochs
    batch      = inputs.load_batch(zip(color, grayscale), batch_size)

    # Train the model and calculate losses.
    print(f"Epoch #{epoch+1} | Batch: {epoch * batch_size} - {(epoch + 1) * batch_size}")
    models, losses = logic.train(models, batch)

    # Save the losses and models in the results directory.
    outputs.save_losses(losses, results_path, epoch + 1)
    outputs.save_models(models, results_path, epoch + 1)

print("Starting best model test.")
test  = inputs.load_dataset(os.path.join(results_path, "test"))
model = inputs.get_best_model(results_path)
logic.test(results_path, model, test)

# Outputs of the models.
outputs.plot_outputs(*losses)