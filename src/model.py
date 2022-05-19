import torch
from torch.nn import (Conv2d, ConvTranspose2d, MaxPool2d, Module, ModuleList,
                      ReLU)
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
from . import config


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Store the convolution and RELU layers
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = ReLU()
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        # Store the encoder blocks and maxpooling layer
        self.encoder = ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        # Initialize an empty list to store the intermediate outputs
        block_outputs = []

        # Loop through the encoder blocks
        for block in self.encoder:
            # Pass the inputs through the current encoder block, store the outputs, and then apply maxpooling on the output
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)

        # Return the list containing the intermediate outputs
        return block_outputs


class Decoder(Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        # Initialize the number of channels, upsampler blocks, and decoder blocks
        self.channels = channels
        self.upconvs = ModuleList([ConvTranspose2d(
            channels[i], channels[i+1], 2, 2) for i in range(len(channels) - 1)])
        self.decoder = ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels) - 1)])

    def forward(self, x, encoder_features):
        # Loop through the number of channels
        for i in range(len(self.channels) - 1):
            # Pass the inputs through upsampler blocks
            x = self.upconvs[i](x)

            # Crop the current features from the encoder blocks, concatenate them with the current upsampled features, and pass the concatenated output through the current decoder block
            feature = self.crop(encoder_features[i], x)
            x = torch.cat([x, feature], dim=1)
            x = self.decoder[i](x)
        # Return the final decoder output
        return x

    def crop(self, feature, x):
        # Grab the dimensions of the inputs, and crop the encoder features to match the dimensions
        (_, _, H, W) = x.shape
        cropped_features = CenterCrop([H, W])(feature)

        # Return the cropped features
        return cropped_features


class UNet(Module):
    def __init__(self, encoder_channels=(3, 64, 128, 256, 512, 1024), decoder_channels=(1024, 512, 256, 128, 64), num_classes=1, retain_dim=True, out_size=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):
        super().__init__()
        # Initialize the encoder and decoder
        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)

        # Initialize the regression head and store the class variables
        self.head = Conv2d(decoder_channels[-1], num_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        # Grab the features from the encoder
        encoder_features = self.encoder(x)

        # Pass the encoder features through the decoder making sure that their dimensions are suited for concatenatation
        decoder_features = self.decoder(
            encoder_features[::-1][0], encoder_features[::-1][1:])

        # Pass the decoder features through the regression head to obtain segmentation mask
        map = self.head(decoder_features)
        # Check to see if we are retaining the original output dimensios and if so, then resize the output to match them
        if self.retain_dim:
            map = F.interpolate(map, self.out_size)
        # Return the segmentation map
        return map
