import os
from torch.utils.data import DataLoader
from src.datasets.dataset import GalaxyDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torchvision.transforms as transforms
import torch
from src.models.vit import get_vit_model
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import collections
import pandas as pd

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def main():
    df = pd.read_csv("data/gz2_three_class_labels.csv")

    print("Dataset size:", len(df))

    start_epoch = 0
    best_acc = 0.0

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"])
    train_df.to_csv("train_labels.csv", index=False)
    val_df.to_csv("val_labels.csv", index=False)

    train_dataset = GalaxyDataset("train_labels.csv", "data/images3_224", train_transform)
    val_dataset = GalaxyDataset("val_labels.csv", "data/images3_224", val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=8, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_vit_model().to(device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "heads" in name or "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": 3e-5},
        {"params": head_params, "lr": 3e-4}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    checkpoint_path = "Models/vit_galaxy3_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        print(f"Resuming from epoch {start_epoch} with best accuracy {best_acc:.2f}%")


    for epoch in range(start_epoch, start_epoch + 25):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print("Epoch:", epoch, "Loss:", total_loss / len(train_loader))
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predicted = outputs.argmax(dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100* correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        cm = confusion_matrix(all_labels, all_preds, normalize="true")
        print("Confusion Matrix:")
        print(cm)
        print("Prediction Distribution:", collections.Counter(all_preds))
        scheduler.step()
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "Models/vit_galaxy3_best.pth")
            print("New best model saved with accuracy: {:.2f}%".format(best_acc))
    
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc
        }, checkpoint_path)
    
    torch.save(model.state_dict(), "Models/vit_galaxy3.pth")

if __name__ == "__main__":
    main()