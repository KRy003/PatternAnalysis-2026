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
from Models.LearningModels import UNetClassifier
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")
print("Using device:", device)
#Configuration
num_epochs = 10
learning_rate = 0.001

# Model Initialisation
model = UNetClassifier().to(device)
# Optimiser Initialisation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Loss function
criterion = nn.CrossEntropyLoss()

'''
Dataset class for loading images and masks from the OASIS dataset.
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)

        self.image_files = sorted(self.image_path.glob("*.png"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Convert image filename to mask filename
        mask_name = img_path.name.replace("case_", "seg_", 1)
        mask_file = self.mask_path / mask_name

        # Load images
        image = Image.open(img_path).convert("L")
        image = self.to_tensor(image)
        mask = Image.open(mask_file).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        mask = torch.from_numpy((mask // 85).astype(np.int64))

        return image, mask

    
print("Loading dataset...")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "OASIS"

# Training dataset
OasisTrainset = Dataset(image_path=DATA_DIR / "keras_png_slices_train",mask_path=DATA_DIR / "keras_png_slices_seg_train")
OasisTrainLoader = torch.utils.data.DataLoader(OasisTrainset, batch_size=16, shuffle=True)

# Validation dataset
OasisValset = Dataset(image_path=DATA_DIR / "keras_png_slices_validate",mask_path=DATA_DIR / "keras_png_slices_seg_validate")
OasisValLoader = torch.utils.data.DataLoader(OasisValset, batch_size=16, shuffle=True)

# Test dataset
OasisTestset = Dataset(image_path=DATA_DIR / "keras_png_slices_test",mask_path=DATA_DIR / "keras_png_slices_seg_test")
OasisTestLoader = torch.utils.data.DataLoader(OasisTestset, batch_size=16,shuffle=False)

print(f"Number of training samples: {len(OasisTrainset)}")
print(f"Number of validation samples: {len(OasisValset)}")
print(f"Number of test samples: {len(OasisTestset)}")
print(f"Batch size: {OasisTrainLoader.batch_size}")
print(f"Image shape: {OasisTrainset[0][0].shape}")
print(f"Mask shape: {OasisTrainset[0][1].shape}")
print(f"Unique mask values: {torch.unique(OasisTrainset[0][1])}")


#
# Training Loop
#
num_classes = 4
@torch.no_grad()
def evaluate_dice(model, loader, device, num_classes=4):
    model.eval()

    intersections = torch.zeros(num_classes, device=device)
    pred_counts = torch.zeros(num_classes, device=device)
    true_counts = torch.zeros(num_classes, device=device)

    for images, masks in loader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        for c in range(num_classes):
            predicted = predictions == c
            actual = masks == c

            intersections[c] += (predicted & actual).sum()
            pred_counts[c] += predicted.sum()
            true_counts[c] += actual.sum()

    # Display the first prediction
    '''
    plt.imshow(predictions[0].cpu(), cmap="viridis", vmin=0, vmax=3)
    plt.title("Predicted Brain Segmentation")
    plt.axis("off")
    plt.show()

    '''
    dice = ((2 * intersections) / (pred_counts + true_counts).clamp(min=1))

    return dice.cpu().tolist()

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    time_start = time.time()

    for images, masks in OasisTrainLoader:
        images = images.to(device).float()
        masks = masks.to(device).long()

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, masks)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()
    time_end = time.time()

    '''
    Printing the loss and validation Dice scores after each epoch.
    '''

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {total_loss / len(OasisTrainLoader):.4f}, "
        f"Time: {time_end - time_start:.2f}s"
    )
    dice_scores = evaluate_dice(model, OasisValLoader, device, num_classes=4)

    print("Validation Dice:")
    for class_id, score in enumerate(dice_scores):
        print(f"  Class {class_id}: {score:.4f}")
    print(f"Mean Dice: {sum(dice_scores) / len(dice_scores):.4f}")

    Test_dice_scores = evaluate_dice(model, OasisTestLoader, device, num_classes=4)
    print("Test Dice:")
    for class_id, score in enumerate(Test_dice_scores):
        print(f"  Class {class_id}: {score:.4f}")
    print(f"Mean Dice: {sum(Test_dice_scores) / len(Test_dice_scores):.4f}")
    
