import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from models.unet import LightweightUNet

class CardHolderDataset(Dataset):
    def __init__(self, split_file, images_dir="dataset/images", masks_dir="dataset/masks", img_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        with open(split_file, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, name)
        mask_path = os.path.join(self.masks_dir, name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size)

        # Normalize to tensors
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_tensor = torch.tensor((mask > 127).astype(np.float32), dtype=torch.float32).unsqueeze(0)

        return img_tensor, mask_tensor, name

def dice_loss(pred_logits, target, smooth=1.0):
    pred = torch.sigmoid(pred_logits)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def train_model(
    epochs=12,
    batch_size=8,
    lr=1e-3,
    save_path="models/cardholder_unet.pth",
    metrics_path="models/eval_metrics.json"
):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_ds = CardHolderDataset("dataset/splits/train.txt")
    val_ds = CardHolderDataset("dataset/splits/val.txt")
    test_ds = CardHolderDataset("dataset/splits/test.txt")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LightweightUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    best_val_iou = 0.0

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = bce(preds, masks) + dice_loss(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = torch.sigmoid(model(imgs)) > 0.5
                target = masks > 0.5
                intersection = (preds & target).float().sum(dim=(1, 2, 3))
                union = (preds | target).float().sum(dim=(1, 2, 3))
                iou = (intersection + 1e-6) / (union + 1e-6)
                val_ious.extend(iou.cpu().tolist())

        mean_val_iou = np.mean(val_ious)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val mIoU: {mean_val_iou:.4f}")

        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            torch.save(model.state_dict(), save_path)

    print(f"\nBest model saved to {save_path} (Val mIoU: {best_val_iou:.4f})")

    # Final Test Split Evaluation (Held-out 10%)
    print("\nEvaluating on held-out TEST split (Zero data leakage)...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_ious, precisions, recalls = [], [], []
    with torch.no_grad():
        for imgs, masks, _ in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = (torch.sigmoid(model(imgs)) > 0.5).float()
            
            tp = (preds * masks).sum(dim=(1, 2, 3))
            fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
            fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))

            iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            prec = (tp + 1e-6) / (tp + fp + 1e-6)
            rec = (tp + 1e-6) / (tp + fn + 1e-6)

            test_ious.extend(iou.cpu().tolist())
            precisions.extend(prec.cpu().tolist())
            recalls.extend(rec.cpu().tolist())

    metrics = {
        "architecture": "LightweightUNet",
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_images_count": len(test_ds),
        "mean_test_mask_iou": float(np.mean(test_ious)),
        "mean_test_precision": float(np.mean(precisions)),
        "mean_test_recall": float(np.mean(recalls)),
        "map_at_50": float(np.mean([1.0 if x >= 0.50 else 0.0 for x in test_ious])),
        "map_50_95": float(np.mean([np.mean([1.0 if x >= t else 0.0 for t in np.arange(0.50, 1.00, 0.05)]) for x in test_ious]))
    }

    print("="*50)
    print("TEST SPLIT EVALUATION METRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("="*50)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_model()
