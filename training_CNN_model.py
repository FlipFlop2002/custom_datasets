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



### creating functions to train and test the model

def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_probs = torch.softmax(y_pred, dim=1)  # probabilities from 0 to 1
        y_pred_class = y_pred_probs.argmax(dim=1)  # predicted label (0/1/2)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_probs = torch.softmax(test_pred_logits, dim=1)  # probabilities from 0 to 1
            test_pred_labels = test_pred_probs.argmax(dim=1)   # predicted label (0/1/2)
            test_acc += (test_pred_labels==y).sum().item()/len(test_pred_labels)

        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)
        return test_loss, test_acc



### create a function to train and test the model 
def train(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               test_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               num_epochs: int):
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        print(f"Epoch: {epoch}\n")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}\n")
        print("############################################################\n")
    
    return results



### train the model
torch.manual_seed(42)
torch.cuda.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data_model_0_simple.classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

EPOCHS = 5
from timeit import default_timer as timer
start_time = timer()


model_0_results = train(model=model_0,
      train_dataloader=train_dataloader_simple,
      test_dataloader=test_dataloader_simple,
      loss_fn=loss_fn,
      optimizer=optimizer,
      device=device,
      num_epochs=EPOCHS)

end_time = timer()
print(f"Total training tim {end_time-start_time:.3f} seconds.")


### plottig the loss curves of model_0
def plot_loss_curves(results: dict[str, list[float]]):
    """Plots training ccurves of a results dictionary"""
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    acc = results["train_acc"]
    test_acc = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="train_accuracy")
    plt.plot(epochs, test_acc, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


plot_loss_curves(model_0_results)


