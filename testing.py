import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score

# Dataset for test images and masks
class TestDataset(Dataset):
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

        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # Grayscale mask

        # Normalize mask to binary (0 or 1)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, self.images[index]

# Function to calculate Jaccard Index (IoU)
def jaccard_index(pred, target):
    # Convert predictions and targets to binary (0 or 1)
    pred = (pred > 0.5).astype(np.float32)  # Applying threshold
    target = (target > 0.5).astype(np.float32)  # Applying threshold

    intersection = np.sum(pred * target)  # Equivalent to bitwise AND
    union = np.sum(pred) + np.sum(target) - intersection  # Equivalent to bitwise OR
    return intersection / union if union != 0 else 0


# Function to save predictions and calculate metrics
def save_test_predictions_and_metrics(loader, model, test_folder, mask_folder, device="cuda"):
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    model.eval()

    loss_fn = BCEWithLogitsLoss()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    all_preds = []
    all_targets = []
    jaccard_values = []  # To store Jaccard Index per image

    with torch.no_grad():
        for x, y, file_names in tqdm(loader, desc="Testing and Saving Results"):
            x = x.to(device)  # Input images
            y = y.to(device).unsqueeze(1)  # Targets with channel dimension

            # Model predictions
            preds = torch.sigmoid(model(x))
            preds_binary = (preds > 0.5).float()

            # Calculate loss
            loss = loss_fn(preds, y)
            total_loss += loss.item()

            # Accuracy calculation
            correct_pixels += (preds_binary == y).sum().item()
            total_pixels += torch.numel(y)

            # Flatten predictions and targets for metric calculation
            all_preds.extend(preds_binary.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

            # Calculate Jaccard Index for the batch (per image)
            batch_jaccard = []
            for i in range(preds_binary.shape[0]):
                pred_flat = preds_binary[i].cpu().numpy().flatten()
                target_flat = y[i].cpu().numpy().flatten()
                jaccard_val = jaccard_index(pred_flat, target_flat)
                batch_jaccard.append(jaccard_val)

            # Append batch Jaccard values
            jaccard_values.extend(batch_jaccard)

            # Save each test image and corresponding mask
            for i in range(preds_binary.shape[0]):  # Iterate through the batch
                test_image_path = os.path.join(test_folder, file_names[i])
                pred_mask_path = os.path.join(mask_folder, file_names[i].replace(".jpg", "_mask.jpg"))
                torchvision.utils.save_image(x[i], test_image_path)
                torchvision.utils.save_image(preds_binary[i], pred_mask_path)

    # Calculate metrics
    f1 = f1_score(all_targets, all_preds, average="binary")
    avg_loss = total_loss / len(loader)
    accuracy = correct_pixels / total_pixels
    avg_jaccard = np.mean(jaccard_values)  # Average Jaccard Index across all images

    print(f"Average Loss: {avg_loss}")
    print(f"F1 Score (Dice Similarity): {f1}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Jaccard Index (IoU): {avg_jaccard:.4f}")


# Main function for testing
def test_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_IMG_DIR = "/content/drive/MyDrive/data/val_images/"
    TEST_MASK_DIR = "/content/drive/MyDrive/data/val_masks/"
    SAVED_TEST_IMAGES_DIR = "/content/drive/MyDrive/test_images_saved/"
    SAVED_PRED_MASKS_DIR = "/content/drive/MyDrive/predicted_masks/"
    CHECKPOINT_FILE = "/content/drive/MyDrive/checkpointss/checkpoint.pth.tar"

    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    test_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    test_ds = TestDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Load model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        print("Model loaded from checkpoint.")
    else:
        raise ValueError(f"Checkpoint file not found at {CHECKPOINT_FILE}")

    # Save test predictions and calculate metrics
    save_test_predictions_and_metrics(
        loader=test_loader,
        model=model,
        test_folder=SAVED_TEST_IMAGES_DIR,
        mask_folder=SAVED_PRED_MASKS_DIR,
        device=DEVICE
    )

if __name__ == "__main__":
    test_model()