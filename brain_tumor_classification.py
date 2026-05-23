# -*- coding: utf-8 -*-
"""
Brain Tumor MRI Classification Pipeline
An end-to-end deep learning workflow using PyTorch to process, train, 
and evaluate MRI image scans for multi-class tumor detection.
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import confusion_matrix

# Global environment configuration
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_seed(seed_value=42):
    """Pin operational seed variance for replicable behavior across executions."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class BrainTumorDataset(Dataset):
    """
    Custom map-style Dataset indexer targeting clinical folder directories.
    Expects dataset subdirectory path to contain category folders:
    e.g., data/raw/Dataset/{glioma, meningioma, pituitary, no_tumor}/*.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("**/*.jpg")) + list(self.root_dir.glob("**/*.jpeg"))
        
        # Extract class names dynamically based on container subfolder naming configurations
        self.classes = sorted(list(set([p.parent.name for p in self.image_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[img_path.parent.name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def build_model(num_classes=4):
    """Instantiates a pre-trained ResNet-18 backbone modified for target class projections."""
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Executes a single optimization pass across the training subset partition."""
    model.train()
    running_loss = 0.0
    correct_counts = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_counts += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_counts / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, device, class_names):
    """Evaluates system validation boundaries and exports a normalized Confusion Matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate performance matrices
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    total_accuracy = np.trace(cm) / np.sum(cm)
    
    print(f"\nEvaluation Performance Absolute Accuracy: {total_accuracy:.4f}")
    
    # Render and save normalized diagnostic confusion map
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, ax=ax, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted Diagnostic Label')
    ax.set_ylabel('True Pathology Label')
    ax.set_title('Normalized Confusion Matrix Heatmap')
    
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("plots/brain_tumor_confusion_matrix.png", dpi=300)
    plt.close()
    print("✓ Model diagnostic matrix heatmap written successfully to plots/")
    return total_accuracy

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target computation engine assigned: {device}")
    
    # Configure project folder directory frameworks
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Image standardization transformation pipelines
    pd_mean = [0.485, 0.456, 0.406]
    pd_std = [0.229, 0.224, 0.225]
    
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pd_mean, std=pd_std)
    ])
    
    # Check for dataset folders
    dataset_path = Path("data/raw/Dataset")
    if not dataset_path.exists():
        print(f"\n[Notice]: Place your category subfolders into '{dataset_path}' to execute training loops.")
        return
        
    print("Loading dataset indices...")
    full_dataset = BrainTumorDataset(root_dir=dataset_path, transform=transform_pipeline)
    
    if len(full_dataset) == 0:
        print(f"No valid JPG/JPEG images found inside '{dataset_path}'. Verification paused.")
        return
        
    # Standard 80/20 cohort sequence split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Instantiate neural layers
    model = build_model(num_classes=4)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Beginning training loops across {train_size} images...")
    epochs = 25
    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1:02d}/{epochs}] -> Loss: {loss:.4f} | Training Accuracy: {acc*100:.2f}%")
        
    print("\nTraining complete. Initiating system evaluation phase...")
    evaluate_model(model, test_loader, device, class_names=full_dataset.classes)

if __name__ == "__main__":
    main()
"""
