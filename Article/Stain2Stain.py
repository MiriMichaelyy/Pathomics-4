#!/bin/python3
import os
import argparse

import logic
import Models
import inputs
import outputs
import preprocess

##############################
# ARGUMENTS & PARAMETERS     #
##############################
parser = argparse.ArgumentParser(description="Normalize a dataset of images.")
parser.add_argument("--dataset",       action="store",      required=True, type=str,                       help="Path to the original dataset.")
parser.add_argument("--channels",      action="store",      default=3,     type=int,                       help="Number of channels in the images.")
parser.add_argument("--image_width",   action="store",      default=256,   type=int,   dest="width",       help="Width of the final images.")
parser.add_argument("--image_height",  action="store",      default=256,   type=int,   dest="height",      help="Height of the final images.")
parser.add_argument("--train",         action="store",      default=0.5,   type=float,                     help="Percentage of the images that go into training.")
parser.add_argument("--test",          action="store",      default=0.5,   type=float,                     help="Percentage of the images that go into testing.")
parser.add_argument("--val",           action="store",      default=0,     type=float, dest="validation",  help="Percentage of the images that go into validation.")
parser.add_argument("--epochs",        action="store",      default=15,    type=int,                       help="Number of epochs, in each train the model with a random batch of images.")
parser.add_argument("--preprocessed",  action="store_true", default=False,                                 help="Preprocess the dataset into color, grayscale & combined and then split into train, test and validation.")
parser.add_argument("--input_format",  action="store",      default="scn", choices=["scn", "png", "tiff"], help="Input images format.")
parser.add_argument("--output_format", action="store",      default="png", choices=["scn", "png", "tiff"], help="Output format to save.")
args = parser.parse_args()

##############################
# INPUT VALIDATION           #
##############################
if not os.path.exists(args.dataset):
    print(f"The dataset path ({args.dataset}) does not exists.")
    exit()

# Normalize the percentages to 100%
total_weight = args.train + args.test + args.validation
if total_weight != 1:
    args.train      /= total_weight
    args.test       /= total_weight
    args.validation /= total_weight

if args.width <= 0 or args.height <= 0:
    print("Image dimensions are non-positive.")
    exit()

if args.channels <= 0:
    print("Invalid number of channels.")
    exit()

if args.epochs <= 0:
    print("Invalid number of epochs.")
    exit()

# Create directories.
processed_path = os.path.join(args.dataset, "Processed")
os.makedirs(processed_path,                                          exist_ok=True)
os.makedirs(os.path.join(processed_path, "color"),                   exist_ok=True)
os.makedirs(os.path.join(processed_path, "grayscale"),               exist_ok=True)
os.makedirs(os.path.join(processed_path, "combined"),                exist_ok=True)
os.makedirs(os.path.join(processed_path, "train",      "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "train",      "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "train",      "combined"),  exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",       "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",       "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",       "combined"),  exist_ok=True)
os.makedirs(os.path.join(processed_path, "test",       "generated"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",        "color"),     exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",        "grayscale"), exist_ok=True)
os.makedirs(os.path.join(processed_path, "val",        "combined"),  exist_ok=True)

results_path = os.path.join(args.dataset, "Results")
os.makedirs(results_path,                         exist_ok=True)
os.makedirs(os.path.join(results_path, "models"), exist_ok=True)

# Compile the parameters into compact variables.
shape = (args.height, args.width, args.channels)
split = (args.train,  args.test,  args.validation)

##############################
# PREPROCESS DATASETS        #
##############################
if not args.preprocessed:
    # Generate basic processed dataset.
    size = 0
    print(f"Splitting original images into {args.width}x{args.height} (color, grayscale & combined).")
    for image in inputs.load_dataset(args.dataset, suffix=args.input_format):
        color, grayscale, combined = preprocess.process_image(image, shape)
        size += outputs.save_dataset(os.path.join(processed_path, "color"),     color,     offset=size, suffix=args.output_format)
        size += outputs.save_dataset(os.path.join(processed_path, "grayscale"), grayscale, offset=size, suffix=args.output_format)
        size += outputs.save_dataset(os.path.join(processed_path, "combined"),  combined,  offset=size, suffix=args.output_format)

    # Load dataset generators.
    print("Loading color & grayscale datasets.")
    color     = inputs.load_dataset(os.path.join(processed_path, "color"),     suffix=args.output_format)
    grayscale = inputs.load_dataset(os.path.join(processed_path, "grayscale"), suffix=args.output_format)
    combined  = inputs.load_dataset(os.path.join(processed_path, "combined"),  suffix=args.output_format)

    # Create sup-datasets.
    print(f"Splitting datasets into train ({args.train}%), test ({args.test}%) and validation ({args.validation}%).")
    train, test, val = preprocess.split_dataset((color, grayscale, combined), size, split)

    # Save processed datasets.
    color, grayscale, combined = train
    outputs.save_dataset(os.path.join(processed_path, "train", "color"),     color,     offset=0, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "train", "grayscale"), grayscale, offset=0, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "train", "combined"),  combined,  offset=0, suffix=args.output_format)
    color, grayscale, combined = test
    outputs.save_dataset(os.path.join(processed_path, "test", "color"),     color,     offset=0, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "test", "grayscale"), grayscale, offset=0, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "test", "combined"),  combined,  offset=0, suffix=args.output_format)
    color, grayscale, combined = val
    outputs.save_dataset(os.path.join(processed_path, "val", "color"),     color,     offset=0, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "val", "grayscale"), grayscale, offset=0, suffix=args.output_format)
    outputs.save_dataset(os.path.join(processed_path, "val", "combined"),  combined,  offset=0, suffix=args.output_format)

##############################
# MAIN LOGIC                 #
##############################
# Train & test the GAN model.
print("Starting to train models.")
color      = inputs.load_dataset(os.path.join(processed_path, "train", "color"),     suffix=args.output_format)
grayscale  = inputs.load_dataset(os.path.join(processed_path, "train", "grayscale"), suffix=args.output_format)
size       = inputs.get_size(os.path.join(processed_path, "train", "color"))
size      += inputs.get_size(os.path.join(processed_path, "test",  "color"))
size      += inputs.get_size(os.path.join(processed_path, "val",   "color"))

models = Models.define_models(shape)
for epoch in range(args.epochs):

    # Split the dataset into equal batches.
    # Convert the image into numpy arrays.
    batch_size = size // args.epochs
    batch      = inputs.load_batch(zip(color, grayscale), batch_size)

    # Train the model and calculate losses.
    print(f"Epoch #{epoch+1} | Batch: {epoch * batch_size} - {(epoch + 1) * batch_size}")
    models, losses = logic.train(models, batch)

    # Save the losses and models in the results directory.
    outputs.save_losses(losses, results_path, epoch + 1)
    outputs.save_models(models, results_path, epoch + 1)

print("Starting best model test.")
test  = inputs.load_dataset(os.path.join(results_path, "test"), suffix=args.output_format)
model = inputs.get_best_model(results_path)
logic.test(results_path, model, test)

# Outputs of the models.
outputs.plot_outputs(*losses)