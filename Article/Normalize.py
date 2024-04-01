#!/bin/python3
import os
import argparse

import numpy

import logic
import inputs
import outputs

##############################
# ARGUMENTS & PARAMETERS     #
##############################
parser = argparse.ArgumentParser(description="Normalize a dataset of images.")
parser.add_argument("--model",        action="store", dest="model",    type=str, required=True,  help="Path to the model for normalization.")
parser.add_argument("--dataset",      action="store", dest="dataset",  type=str, required=True,  help="Path to the dataset to normalize.")
parser.add_argument("--results",      action="store", dest="results",  type=str, required=False, help="Path to store the normalized dataset.")
parser.add_argument("--channels",     action="store", dest="channels", type=int, default=3,      help="Number of channels in the images.")
parser.add_argument("--image_width",  action="store", dest="width",    type=int, default=256,    help="Width of the images in the dataset.")
parser.add_argument("--image_height", action="store", dest="height",   type=int, default=256,    help="Height of the images in the dataset.")
args = parser.parse_args()

##############################
# INPUT VALIDATION           #
##############################
if not os.path.exists(args.model):
    print(f"The model path ({args.model}) does not exists.")
    exit()

if not os.path.exists(args.dataset):
    print(f"The dataset path ({args.dataset}) does not exists.")
    exit()

if "results" not in args:
    args.results = os.path.join(args.dataset, "normalized")

##############################
# MAIN LOGIC                 #
##############################
# model   = inputs.get_best_model(args.model)
class StubModel:
    def predict(self, image):
        return image

model   = StubModel()
dataset = inputs.load_dataset(args.dataset)
for i, image in enumerate(dataset):
    parsed_image = numpy.array(image, dtype=float)
    normalized_image = logic.normalize(model, parsed_image)
    outputs.save_original(args.results, normalized_image)