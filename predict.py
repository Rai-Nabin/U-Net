from configparser import Interpolation
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config


def prepare_plot(original_image, original_mask, prediction_mask):

    # Initialize figure
    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

    # Plot the images
    ax[0].imshow(original_image)
    ax[1].imshow(original_mask)
    ax[2].imshow(prediction_mask)

    # Set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # Set the layout of the figure and display it
    plt.tight_layout()
    plt.show()


def make_predictions(model, image):
    # Set model to evaluation mode
    model.eval()
    # Turn off gradient tracking
    with torch.no_grad():

        image = image.astype("float32")/255.0

        # Make the channel axis to be the leading one, add a batch dimension, create a pyTorch tensor, and flash it to the current device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(config.DEVICE)

        # Make the prediction, pass the results through the sigmoid function, and convert the result to a numpy array
        prediction_mask = model(image).squeeze()
        prediction_mask = torch.sigmoid(prediction_mask)
        prediction_mask = prediction_mask.cpu().numpy()

        # # Filter out the weak predictions and convert them to integers
        # prediction_mask = (prediction_mask > config.THRESHOLD) * 255
        # prediction_mask = prediction_mask.astype(np.uint8)

        return prediction_mask


def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


def main():
    # Load the image path from test file and randomly select 1 image paths
    print("[INFO]: Loading up test image paths...")
    image_paths = open(config.TEST_PATH).read().strip().split("\n")
    image_paths = np.random.choice(image_paths, size=2)

    # Load model from disk and flash it to the current device
    print("[INFO]: Loading up model ...")

    unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
    

    # Iterate over the randomly selected test images paths
    for image_path in image_paths:
        # Find the filename and generate the path to ground truth
        file_name = image_path.split(os.path.sep)[-1]
        ground_truth_path = os.path.join(
            config.MASK_DATASET_PATH, os.path.splitext(file_name)[0]+'.png')

        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Resize the image and make a copy of it for visualization
        image = cv.resize(image, (572, 572))

        original_image = image.copy()

        # Make predictions and visualize the result
        predicted_mask = make_predictions(unet, image)
        predicted_mask = preprocess_mask(predicted_mask)

        # Load the ground truth segmentation mask and resize it
        ground_truth_mask = cv.imread(ground_truth_path, cv.IMREAD_UNCHANGED)
        ground_truth_mask = cv.resize(
            ground_truth_mask, (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))

        ground_truth_mask = preprocess_mask(ground_truth_mask)

        prepare_plot(original_image, ground_truth_mask, predicted_mask)


if __name__ == "__main__":
    main()
