from cProfile import label
import os
import time

import matplotlib.pyplot as plt
import torch
from imutils import paths
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src import config
from src.dataset import SegmentationDataset
from src.model import UNet

# Load the image and mask filepaths in a sorted manner
image_paths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
mask_paths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))

# Partition the data into training and testing splits using 85% of the data for training and the remaining 15% for testing
split = train_test_split(image_paths, mask_paths,
                         test_size=config.TEST_SPLIT, random_state=42)

# Unpack the data split
train_images, test_images = split[:2]
train_masks, test_masks = split[2:]

# Write the testing imager paths to disk so that we can use them when evaluaing/testing our model
print("[INFO]: Saving testing image paths...")
with open(config.TEST_PATHS, "w") as f:
    f.write("\n".join(test_images))

# Define transformations
transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(
    (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), transforms.ToTensor()])

# Create the train and test datasets
train_datasets = SegmentationDataset(
    image_paths=train_images, mask_paths=train_masks, transforms=transforms)
test_datasets = SegmentationDataset(
    image_paths=test_images, mask_paths=test_masks, transforms=transforms)

print(f'[INFO]: Found {len(train_datasets)} examples in the training set...')
print(f'[INFO]: Found {len(test_datasets)} examples in the test set...')

# Create the training and test data loaders
train_loader = DataLoader(train_datasets, shuffle=True, batch_size=config.BATCH_SIZE,
                          pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())
test_loader = DataLoader(test_datasets, shuffle=False, batch_size=config.BATCH_SIZE,
                         pin_memory=config.PIN_MEMORY, num_workers=os.cpu_count())


# Initialze the UNet model
unet_model = UNet().to(config.DEVICE)

# Initialize loss function and optimizer
loss_function = BCEWithLogitsLoss()
optimizer = Adam(unet_model.parameters(), lr=config.INIT_LR)

# Calculate steps per epoch for training and test set
train_steps = len(train_datasets) // config.BATCH_SIZE
test_steps = len(test_datasets) // config.BATCH_SIZE

# Initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

# Loop over epochs
print("[INFO]: Training the netwrok...")
star_time = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
    # Set the model in training mode
    unet_model.train()
    # Initialize the total training and validation loss
    total_train_loss = 0
    total_test_loss = 0

# Loop over the training set
for (i, (x, y)) in enumerate(train_loader):
    # Send the input to the device
    (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

    # Perform a forward pass and calculate the training loss
    prediction = unet_model(x)
    loss = loss_function(prediction, y)

    # First, zero out any previously accumulated gradients, then perform backpropagation, and then update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Add the loss to the total training loss so far
    total_train_loss += loss

    # Switch off to autograd
    with torch.no_grad():
        # Set the model in evaluation  mode
        unet_model.eval()

        # Loop over the validation set
        for (x, y) in test_loader:
            # Send the input to the device
            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
            # Make the predictions and calculate the validation loss
            prediction = unet_model(x)
            total_test_loss += loss_function(prediction, y)

    # Calculate the average training and validation loss
    average_train_loss = total_train_loss / train_steps
    average_test_loss = total_test_loss / test_steps

    # Update training history
    H["train_loss"].append(average_train_loss.cpu().detach().numpy())
    H["test_loss"].append(average_test_loss.cpu().detach().numpy())

    # Print the model training and validation loss information
    print("[INFO]: EPOCH: {}/{}".format(e+1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Test loss: {:.4f}".format(
        average_train_loss, average_test_loss))

    # Display the total time needed to perform the training
    end_time = time.time()
    print("[INFO]: Total time taken to train the model: {:.2f}s".format(
        end_time-star_time))


# Plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss",], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)

# Serialize the model to disk
torch.save(unet_model, config.MODEL_PATH)