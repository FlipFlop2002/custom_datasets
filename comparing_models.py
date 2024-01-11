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

# loading models results from json files
import json
with open('models_results/model_0_results.json', 'r') as jh:
    model_0_results = json.load(jh)
with open('models_results/model_1_results.json', 'r') as jh:
    model_1_results = json.load(jh)


# compare models
import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

print(model_0_df)

# setup a plot
plt.figure(figsize=(15, 10))
epochs = range(len(model_0_df))

plt.subplot(2, 2, 1)
plt.plot(epochs, model_0_df['train_loss'], label="Model 0")
plt.plot(epochs, model_1_df['train_loss'], label="Model 1")
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, model_0_df['train_acc'], label="Model 0")
plt.plot(epochs, model_1_df['train_acc'], label="Model 1")
plt.title("Train Accuracy")
plt.xlabel("Epochs")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, model_0_df['test_loss'], label="Model 0")
plt.plot(epochs, model_1_df['test_loss'], label="Model 1")
plt.title("Test Loss")
plt.xlabel("Epochs")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, model_0_df['test_acc'], label="Model 0")
plt.plot(epochs, model_1_df['test_acc'], label="Model 1")
plt.title("Test Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()


