import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pandas as pd
import os
import numpy as np

# Configuration
BATCH_SIZE = 30
EPOCHS = 20
LEARNING_RATE = 0.0005
NUM_CLASSES = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
OUTPUT_DIR = "result"

# Transform
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


val_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Datasets
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# Mixup
def mixup_data(x, y, alpha=0.6):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# SE 
class SE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.global_avg_pool(x).view(b, c)
        se = self.fc(se).view(b, c, 1, 1)
        return x * se

# ResNext50
model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
model.layer4 = nn.Sequential(
    model.layer4,
    SE(2048)
)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(DEVICE)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=EPOCHS)

best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images, labels_a, labels_b, lam = mixup_data(images, labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (lam * predicted.eq(labels_a).sum().item() + (1 - lam) * predicted.eq(labels_b).sum().item())
    
    train_acc = 100. * correct / total
    scheduler.step()
    print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%')
    
    # Val
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_acc = 100. * correct / total
    print(f'Validation Accuracy: {val_acc:.2f}%')
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pt'))
        print('Model saved!')

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt')))
model.eval()

test_images = sorted(os.listdir(os.path.join(DATA_DIR, 'test')))
data = []

with torch.no_grad():
    for img_name in test_images:
        img_path = os.path.join(DATA_DIR, 'test', img_name)
        image = datasets.folder.default_loader(img_path)
        image = val_transform(image).unsqueeze(0).to(DEVICE)
        output = model(image)
        _, predicted = output.max(1)
        predicted_class = idx_to_class[predicted.item()]
        image_name = os.path.splitext(img_name)[0]
        data.append([image_name, predicted_class])

# Save CSV
pd.DataFrame(data, columns=['image_name', 'pred_label']).to_csv(os.path.join(OUTPUT_DIR, 'prediction.csv'), index=False)