# -*- coding: utf-8 -*-
"""AI(UNET).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13L1vxyzJ9Xdb3QTai0G6vgeVPRXKo3Q4
"""

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define paths for saved_images and checkpoint folders
SAVED_IMAGES_DIR = "/content/drive/My Drive/saved_images"
CHECKPOINT_DIR = "/content/drive/My Drive/checkpointss"

# Create directories if they do not exist
os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# UNET model definition
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, kin_channels, ernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# Dataset class
class MyData(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise ValueError(f"Missing file: {img_path} or {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

# Utility functions
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, filename)
    print(f"=> Saving checkpoint to {checkpoint_path}")
    torch.save(state, checkpoint_path)

def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=2, pin_memory=True):
    train_ds = MyData(image_dir=train_dir, mask_dir=train_maskdir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_ds = MyData(image_dir=val_dir, mask_dir=val_maskdir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += preds.numel()
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Accuracy: {num_correct / num_pixels * 100:.2f}%, Dice Score: {dice_score / len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Save each prediction separately
        for i in range(preds.shape[0]):  # Iterate through the batch
            torchvision.utils.save_image(preds[i], f"{folder}/pred_{idx * preds.shape[0] + i}.png")
            torchvision.utils.save_image(y[i].unsqueeze(0), f"{folder}/true_{idx * preds.shape[0] + i}.png")
    model.train()


# Training function
def train_fn(loader, model, optimizer, loss_fn, scaler, device, epoch, max_epochs):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward
        with torch.amp.autocast(device_type=device):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 37
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    TRAIN_IMG_DIR = "/content/drive/My Drive/data/train_images/"
    TRAIN_MASK_DIR = "/content/drive/My Drive/data/train_masks/"
    VAL_IMG_DIR = "/content/drive/My Drive/data/val_images/"
    VAL_MASK_DIR = "/content/drive/My Drive/data/val_masks/"
    SAVED_IMAGES_DIR = "/content/drive/My Drive/saved_images/"
    CHECKPOINT_DIR = "/content/drive/My Drive/checkpointss/"
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pth.tar")

    # Create folders in Google Drive
    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    train_loader, val_loader = get_loaders(
        train_dir=TRAIN_IMG_DIR, train_maskdir=TRAIN_MASK_DIR,
        val_dir=VAL_IMG_DIR, val_maskdir=VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        epoch_start = load_checkpoint(checkpoint, model, optimizer) + 1
    else:
        epoch_start = 0

    for epoch in range(epoch_start, NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE, epoch, NUM_EPOCHS)
        check_accuracy(val_loader, model, device=DEVICE)
        save_predictions_as_imgs(val_loader, model, folder=SAVED_IMAGES_DIR, device=DEVICE)

        scheduler.step()

        # Save checkpoint
        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch})

if __name__ == "__main__":
    main()

