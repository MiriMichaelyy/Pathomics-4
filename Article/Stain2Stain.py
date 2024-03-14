import os
import sys

import logic
import Models
import inputs
import outputs

if len(sys.argv) <= 1:
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
epochs        = 15

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
if not os.path.exists(f"{results_path}/results"):
    os.makedirs(f"{results_path}/results")
if not os.path.exists(f"{results_path}/models"):
    os.makedirs(f"{results_path}/models")

##############################
# MAIN LOGIC                 #
##############################
num_of_samples = 0
for image in inputs.load_dataset(dataset_path):
    color, grayscale, combined = logic.process_image(image, img_shape)
    num_of_samples += len(color)

    outputs.save_dataset(f"{results_path}/color",     color)
    outputs.save_dataset(f"{results_path}/combined",  combined)
    outputs.save_dataset(f"{results_path}/grayscale", grayscale)

color     = inputs.load_dataset(f"{dataset_path}/color")
combined  = inputs.load_dataset(f"{dataset_path}/combined")
grayscale = inputs.load_dataset(f"{dataset_path}/grayscale")

datasets         = (color, grayscale, combined)
train, test, val = logic.split_dataset(datasets, num_of_samples, dataset_split)

outputs.save_dataset(f"{results_path}/train", train)
outputs.save_dataset(f"{results_path}/test",  test)
outputs.save_dataset(f"{results_path}/val",   val)

# Train & test the GAN model.
print("Training Models")
models = Models.define_models(img_shape)
for epoch in range(epochs):
    print(f"Epoch #{epoch+1}")
    train          = inputs.load_dataset(f"{dataset_path}/train")
    models, losses = logic.train(models, train)
    outputs.save_losses(losses, results_path, epoch + 1)
    outputs.save_models(models, results_path, epoch + 1)

print("Testing...")
test = inputs.load_dataset(f"{dataset_path}/test")
logic.test(inputs.get_best_model(results_path), test)

# Outputs of the models.
outputs.plot_outputs('seaborn')