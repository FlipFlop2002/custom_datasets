import os
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, List
import random
import matplotlib.pyplot as plt

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
image_path_list = list(image_path.glob("*/*/*.jpg"))
train_dir = image_path / "train"
test_dir = image_path / "test"


### helper function to get class names
target_directory = train_dir

class_names_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
print(class_names_found)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the calss folder names in a target directory.
    """
    classes = sorted(entry.name for entry in list(os.scandir(directory)) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Could not find any classes in {directory}... Please check file structure.")
    
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx

print(find_classes(target_directory))



### creating our own dataset
class ImageFolderCustom(Dataset):
    def __init__(self,
                 targ_dir: str,
                 transform=None):
        # super().__init__()
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes = sorted(entry.name for entry in list(os.scandir(targ_dir)) if entry.is_dir())
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns ona sample of data -> (X, y)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


train_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

# testing ImageFolderCustom
train_data_custom = ImageFolderCustom(targ_dir=target_directory,
                                      transform=train_transforms,
                                      )

test_data_custom = ImageFolderCustom(targ_dir=target_directory,
                                      transform=test_transforms,
                                      )

print(train_data_custom, test_data_custom)
print(train_data_custom.classes)
print(train_data_custom.class_to_idx)


### create a function to display random images
def display_random_image(dataset: torch.utils.data.Dataset,
                         classes: List[str]=None,
                         n: int=10,
                         display_shape: bool=True,
                         seed: int=None):
    if n > 10:
        n = 10
        display_shape = False
    
    if seed:
        random.seed(seed)

    random_samples_idx = random.sample(range(len(dataset)), k=n)

    plt.figure(figsize=(16, 4))

    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        targ_image_adjust = targ_image.permute(1, 2, 0)

        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis(False)
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nShape: {targ_image_adjust.shape}"

        plt.title(title)
    plt.show()


display_random_image(train_data_custom,
                     n=12,
                     classes=class_names_found,
                     seed=None)


### turning data into DataLoaders
BATCH_SIZE = 32
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)

img_custom, label_custom = next(iter(train_dataloader_custom))

print(img_custom.shape)
print(label_custom.shape)


# data augmentation -   looking at the same image but from different perspective to artifically increase 
#                       the diversity of a dataset

# we will look at trivialaugment
train_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# plot random transformed images
from images_into_tensors import plot_transformed_images
plot_transformed_images(image_paths=image_path_list,
                        transform=train_transform,
                        n=3,
                        seed=None)
