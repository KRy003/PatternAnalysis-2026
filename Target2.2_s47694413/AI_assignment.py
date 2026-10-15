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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is not available. Using CPU.")

#Configuration
num_epochs = 10
learning_rate = 0.001

# Model Initialisation
model = UNetClassifier().to(device)
# Optimiser Initialisation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Loss function
criterion = nn.BCEWithLogitsLoss()

#
# Data Loading
# 
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

        if not mask_file.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_file}"
            )

        # Load images
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_file).convert("L")

        # Convert to tensors
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        return image, mask

    
print("Loading dataset...")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "OASIS"

# Training dataset
OasisTrainset = Dataset(
    image_path=DATA_DIR / "keras_png_slices_train",
    mask_path=DATA_DIR / "keras_png_slices_seg_train"
)

OasisTrainLoader = torch.utils.data.DataLoader(
    OasisTrainset,
    batch_size=16,
    shuffle=True
)

# Validation dataset
OasisValset = Dataset(
    image_path=DATA_DIR / "keras_png_slices_validate",
    mask_path=DATA_DIR / "keras_png_slices_seg_validate"
)

OasisValLoader = torch.utils.data.DataLoader(
    OasisValset,
    batch_size=16,
    shuffle=False
)

# Test dataset
OasisTestset = Dataset(
    image_path=DATA_DIR / "keras_png_slices_test",
    mask_path=DATA_DIR / "keras_png_slices_seg_test"
)

OasisTestLoader = torch.utils.data.DataLoader(
    OasisTestset,
    batch_size=16,
    shuffle=False
)

print("Training samples:", len(OasisTrainset))
print("Validation samples:", len(OasisValset))
print("Test samples:", len(OasisTestset))

#
# Training Loop
#
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for images, masks in OasisTrainLoader:
        images = images.to(device).float()
        masks = masks.to(device).float()

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

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {total_loss / len(OasisTrainLoader):.4f}"
    )
