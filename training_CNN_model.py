import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, List
import random
import matplotlib.pyplot as plt
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


# replicating TinyVGG architecture (CNN)
### creating transforms and loading data fot Model 0
simple_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

from torchvision import datasets
train_data_model_0_simple = datasets.ImageFolder(root=train_dir,
                                  transform=simple_transform)

test_data_model_0_simple = datasets.ImageFolder(root=test_dir,
                                 transform=simple_transform)

print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(train_data_model_0_simple[0])

# creating DataLoader for train and test data
BATCH_SIZE = 32
train_dataloader_simple = DataLoader(dataset=train_data_model_0_simple,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader_simple = DataLoader(dataset=test_data_model_0_simple,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


print(train_dataloader_simple)

train_images_batch, train_labels_batch = next(iter(train_dataloader_simple))
print(train_images_batch.shape)
print(train_labels_batch[0])
print(train_labels_batch)


### create TinyVGG model class
class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from CNN Explainer
    """
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*13*13,
                      out_features=output_shape)
        )

    
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        # print(x)
        return x
    
class_names = train_data_model_0_simple.classes

torch.manual_seed(42)
model_0 = TinyVGG(3, 10, len(class_names)).to(device)
print(model_0)

image_batch, label_batch = next(iter(train_dataloader_simple))

model_0(image_batch.to(device))
print("########################################")

# using torchinfo to get the info of shapes while data goes through the model
from torchinfo import summary
summary(model_0, input_size=[32, 3, 64, 64])


# creating functions to train and test the model