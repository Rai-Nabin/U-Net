from src import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import os

def prepare_plot(original_image, original_mask, prediction_mask):
    # Initialize figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

    # Plot the images
    ax[0].imshow(original_image)
    ax[1].imshow(original_mask)
    ax[2].imshow(prediction_mask)

    # Set the tiles
    # Set the layout of the figure and display it

def make_predictions(model, image_path):
    # Set model to evaluation mode
    model.eval()
    # Turn off gradient tracking
    with torch.no_grad():
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = image.astype("float32")/255.0

        # Resize the image and make a copy of it for visualization
        image = cv.resize(image, (128, 128))
        original_image = image.copy()

        # Find the filename and generate the path to ground truth
        file_name = image_path.split(os.path.sep)[-1]
        ground_truth_path = os.path.join(config.MASK_DATASET_PATH, file_name)