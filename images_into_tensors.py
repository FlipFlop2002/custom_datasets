import torch
from torch import nn
import requests
import zipfile
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
image_path_list = list(image_path.glob("*/*/*.jpg"))

# jpg -> tensor
data_transform = transforms.Compose([
    #resize our images
    transforms.Resize(size=(64, 64)),
    #flip  the images randomly on the horiontal
    transforms.RandomHorizontalFlip(p=0.5),
    #image into torch.Tensor
    transforms.ToTensor()
])

def plot_transformed_images(image_paths: list, transform: transforms.Compose, n=3, seed=None):
    """
    Selects random images from a path of images and loads/transforms
    them. Then plots the original vs the transformed one.
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as fh:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(fh)
            ax[0].set_title(f"Original\nSize: {fh.size}")
            ax[0].axis(False)

            transformed_img = transform(fh).permute(1, 2, 0)
            ax[1].imshow(transformed_img)
            ax[1].set_title(f"Transformed\nSize: {transformed_img.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show()
            


plot_transformed_images(image_path_list, data_transform)

### setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"



### option 1: Loading image data using 'ImageFolder'

from torchvision import datasets

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(train_data, test_data)

class_names = train_data.classes
print(class_names)
class_dict = train_data.class_to_idx
print(class_dict)

print(train_data[0][0].shape)

plt.figure(figsize=(10, 7))
plt.imshow(train_data[0][0].permute(1, 2, 0))
plt.axis(False)
plt.title(class_names[train_data[0][1]], fontsize=14)
plt.show()


# creating DataLoader for train and test data
BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

img, label = next(iter(train_dataloader))
print(img)
print("-----------------")
print(label)



