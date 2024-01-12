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
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

### download custom image
import requests
custom_image_path = Path("data/04-pizza-dad.jpeg")

# if not custom_image_path.is_file():
#     with open(custom_image_path, "wb") as fh:
#         request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/blob/main/images/04-pizza-dad.jpeg")
#         print("Downloading image...")
#         fh.write(request.content)
# else:
#     print("Image already exists")


### loading custom image with pytorch
custom_img_uint8 = torchvision.io.read_image(str(custom_image_path))
# plt.imshow(custom_img_uint8.permute(1, 2, 0))
# plt.show()

print(custom_img_uint8.shape)
print(custom_img_uint8.dtype)

transform = transforms.Compose([
    transforms.Resize(size=(64, 64))
])

custom_img_uint8_resized = transform(custom_img_uint8)
# plt.imshow(custom_img_uint8_resized.permute(1, 2, 0))
# plt.show()

custom_img = torchvision.io.read_image(str(custom_image_path)).type(torch.float32) / 255
custom_img_resized = transform(custom_img)
print(custom_img_resized.dtype)
print(custom_img_resized.shape)

plt.imshow(custom_img_resized.permute(1, 2, 0))
plt.title("Float32 size 64x64")
plt.show()

print(custom_img_resized.unsqueeze(dim=0).shape)

from TinyVGG_model import TinyVGG
model_1 = TinyVGG(3, 10, 3).to(device)
model_1.load_state_dict(torch.load("models/model_1.pth"))
model_1.eval()
with torch.inference_mode():
    y_logits = model_1(custom_img_resized.unsqueeze(dim=0).to(device))

y_pred_label = torch.argmax(y_logits)
classes = ['pizza', 'steak', 'sushi']
print(f"Predicted dish: {classes[y_pred_label]}")


## next image - sushi
sushi_img = torchvision.io.read_image(str("data/sushi3.jpg")).type(torch.float32) / 255
sushi_img = transform(sushi_img)
print(sushi_img.dtype)
print(sushi_img.shape)

plt.imshow(sushi_img.permute(1, 2, 0))
plt.title("Float32 size 64x64 sushi")
plt.show()


model_1.eval()
with torch.inference_mode():
    y_logits_2 = model_1(sushi_img.unsqueeze(dim=0).to(device))

y_pred_label_2 = torch.argmax(y_logits_2)
print(f"Predicted dish: {classes[y_pred_label_2]}")



