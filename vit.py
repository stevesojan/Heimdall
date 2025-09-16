import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt


def parse_pattern(filename):
    # Remove extension
    base = filename.split('.png')[0]
    # Remove 'CDL' prefix
    if base.startswith("CDL"):
        base = base[3:]
    # Remove trailing number and underscore
    pattern = "_".join(base.split("_")[:-1])
    return pattern


IMG_DIR = 'candlestick_images'

# Dynamically build ALL_PATTERNS from files in the folder
all_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
patterns_set = set()
for fname in all_files:
    pat = parse_pattern(fname)
    patterns_set.add(pat)
ALL_PATTERNS = sorted(list(patterns_set))

PATTERN_TO_IDX = {pat: i for i, pat in enumerate(ALL_PATTERNS)}
IDX_TO_PATTERN = {i: pat for pat, i in PATTERN_TO_IDX.items()}
NUM_CLASSES = len(ALL_PATTERNS)
print("Detected patterns:", ALL_PATTERNS)
print(NUM_CLASSES)

class CandlestickImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.labels = [self.encode_labels(f) for f in self.img_files]

    def encode_labels(self, filename):
        pattern_name = parse_pattern(filename)
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        if pattern_name in PATTERN_TO_IDX:
            label[PATTERN_TO_IDX[pattern_name]] = 1.
        return label

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.img_files)


BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CandlestickImageDataset(IMG_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize Vision Transformer model
vit = models.vision_transformer.vit_b_16(weights=None)
vit.heads[0] = torch.nn.Linear(vit.heads[0].in_features, NUM_CLASSES)

print(vit.heads)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(vit.parameters(), lr=1e-4)
num_epochs = 30

for epoch in range(num_epochs):
    vit.train()
    total_loss = 0.0
    for images, labels in dataloader:
        outputs = vit(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader)}")

torch.save(vit.state_dict(), 'vit_candlestick.pth')


# --- Inference and Annotation ---
def annotate_patterns_on_image(model, img_path, threshold=0.5):
    image_pil = Image.open(img_path).convert("RGB")
    image = transform(image_pil).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)[0]
        pred_idxs = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
        pred_labels = [IDX_TO_PATTERN[idx] for idx in pred_idxs]
    plt.imshow(np.array(image_pil))
    for i, label in enumerate(pred_labels):
        plt.text(10, 30 + i * 40, f"Detected: {label}", color='red', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.7))
    plt.axis('off')
    plt.show()

# Usage example:
# annotate_patterns_on_image(vit, "candlestick_images/CDLHIKKAKE_bullish_4783.png", threshold=0.5)

#%%



#%%