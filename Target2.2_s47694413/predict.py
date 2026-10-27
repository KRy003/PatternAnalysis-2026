import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import time
import matplotlib.pyplot as plt
from modules import UNetClassifier
from dataset import Dataset
from dataset import get_dataloaders
import numpy as np
import nibabel as nib
import torch.nn.functional as F

''' Function Declarations '''
def get_weights_path():
    """
    Returns the path to the weights file.
    """
    return Path(__file__).resolve().parent / "unet_normal_model.pth"

def load_model(model, path, device):
    """
    Loads the model's state dictionary from the specified path.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")

def show_predictions(image, mask, prediction):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image[0], cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='nipy_spectral')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='nipy_spectral')
    plt.title('Prediction')
    plt.axis('off')

def run_weights_on_data(dataloader, device, model, num_classes, intersections, pred_counts, mask_counts):
    model.eval()
    total_inference_time = 0.0
    total_images = 0
    with torch.inference_mode():
        for i, (images, masks) in enumerate(dataloader):

            images = images.to(device).float()
            masks = masks.to(device).long()

            # Ensure previous GPU operations are complete
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            # Forward pass
            logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_inference_time += end - start
            total_images += images.size(0)

            # Convert logits to predicted classes
            predictions = torch.argmax(logits, dim=1)

            # Accumulate Dice statistics
            for class_id in range(num_classes):
                pred_class = predictions == class_id
                mask_class = masks == class_id

                intersections[class_id] += (
                    pred_class & mask_class
                ).sum()

                pred_counts[class_id] += pred_class.sum()
                mask_counts[class_id] += mask_class.sum()

            # Visualize only the first two batches
            if i < 2:
                images_np = images.cpu().numpy()
                masks_np = masks.cpu().numpy()
                predictions_np = predictions.cpu().numpy()

                show_predictions(
                    images_np[0],
                    masks_np[0],
                    predictions_np[0]
                )

    # Average inference latency per image
    average_time = total_inference_time / total_images

    print(
        f"Average inference time per image: "
        f"{average_time * 1000:.2f} ms"
    )
        

'''
############ Main Code ############
'''
''' Variable Declaration'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 6
model = UNetClassifier(in_channels=1, out_channels=6).to(device)

model.load_state_dict(torch.load(get_weights_path(), map_location=device, weights_only=True))
model.eval()

hip_train_loader, hip_val_loader, hip_test_loader, hip_train_set, hip_val_set, hip_test_set = get_dataloaders()
# Statistics declarations
intersections = torch.zeros(num_classes, device=device)
pred_counts = torch.zeros(num_classes, device=device)
mask_counts = torch.zeros(num_classes, device=device)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

''' 
Running Weights 
'''
with torch.inference_mode():
    run_weights_on_data(hip_test_loader, device, model, num_classes, intersections, pred_counts, mask_counts)

plt.show()

'''
Statistics Section
'''
denominators = pred_counts + mask_counts
dice_scores = torch.full((num_classes,), float("nan"), device=device)
valid_classes = denominators > 0
dice_scores[valid_classes] = (2 * intersections[valid_classes] / denominators[valid_classes])

''' Dice Scores '''
# Print individual class scores
print("\n===== Dice Scores =====")
for class_id, score in enumerate(dice_scores):
    print(f"Dice {class_id}: {score.item():.4f}")

# Calculate average Dice score
average_dice = dice_scores[valid_classes].mean().item()
print(f"\nAverage Dice Score: {average_dice:.4f}")
print(f"Average Dice Percentage: {average_dice * 100:.2f}%")

''' VRam Usage '''
torch.cuda.synchronize()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

print(f"Peak VRAM: {peak_vram_mb:.2f} MB")