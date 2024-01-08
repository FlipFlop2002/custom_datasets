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
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
train_dir = image_path / "train"
test_dir = image_path / "test"

train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_simple = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

from torchvision import datasets
train_data_augmented = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform_trivial)

test_data_simple = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform_simple)

BATCH_SIZE = 32
torch.manual_seed(42)
train_dataloader_augmented = DataLoader(dataset=train_data_augmented,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader_simple = DataLoader(dataset=test_data_simple,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


from TinyVGG_model import TinyVGG
model_1 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data_augmented.classes)).to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

NUM_EPOCHS = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

from train_test_functions import *

from timeit import default_timer as timer
start_time = timer()


model_1_results = train(model=model_1,
      train_dataloader=train_dataloader_augmented,
      test_dataloader=test_dataloader_simple,
      loss_fn=loss_fn,
      optimizer=optimizer,
      device=device,
      num_epochs=NUM_EPOCHS)

end_time = timer()
print(f"Total training time {end_time-start_time:.3f} seconds.")

plot_loss_curves(model_1_results)
