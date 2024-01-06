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

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    """
    for dirpath, dirnames, filenames  in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)


### setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"


### visualize an image
random.seed(42)

# get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))
print(image_path_list)


# pick a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)


# get imag ecalss from path name
image_class = random_image_path.parent.stem
print(image_class)

# open image
img = Image.open(random_image_path)

# print metadata
print(f"img height: {img.height}")
print(f"img width: {img.width}")
print("-----------------------")
img.show()


# open img using matplotlib
img_as_array = np.asarray(img)
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"img shape: {img_as_array.shape} ->  [h, w, channels] ")
plt.axis(False)
plt.show()


