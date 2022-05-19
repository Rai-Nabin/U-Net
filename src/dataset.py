import cv2 as cv
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        # Return the number of total samples contained in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Grab the image path from the current index
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the image from disk, swap its channels from BGR to RGB
        image = cv.imread(image_path)
        # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Load the mask from disk 
        mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

        # Check to see if we are applying any transformations
        if self.transforms is not None:
            # Apply the transfomations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)

        # Return a tuple of the image and its mask
        return (image, mask)
