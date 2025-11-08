import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from PIL import Image
import pandas as pd
import numpy as np
import timm

# =====================================================
# "Borrowed" from https://github.com/dstrick17/DacNet/blob/main/scripts/dacnet.py
# =====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_directory, class_names, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_directory = image_directory
        self.transform = transform
        self.class_names = [c.lower().replace(" ", "_") for c in class_names]
        self.class_name_to_index = {name: i for i, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = os.path.join(self.image_directory, row["Image Index"])
        image = Image.open(image_path).convert("RGB")

        label_tensor = torch.zeros(len(self.class_names))
        for label in row["Finding Labels"].split("|"):
            key = label.strip().lower().replace(" ", "_")
            if key in self.class_name_to_index:
                label_tensor[self.class_name_to_index[key]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label_tensor



def get_optimal_thresholds(labels, preds):
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_thresh = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_thresh)
    return thresholds
# =====================================================
# =====================================================


class DenseNetViT(nn.Module):
    def __init__(self, num_classes=14, freeze_backbone=False, freeze_vit=False, use_custom_proj=False):
        super().__init__()
        self.use_custom_proj = use_custom_proj

        # DenseNet backbone
        self.backbone = timm.create_model("densenet121", pretrained=True, features_only=True)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ViT
        # We replace the ViT patch embedding with DenseNet features
        # The positional embeddings are sliced to match the number of tokens
        # 1 CLS token + 49 patch tokens from DenseNet = 50 tokens.
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()   # remove original classification head
        self.vit.patch_embed = nn.Identity()
        self.vit.pos_embed = nn.Parameter(self.vit.pos_embed[:, :50, :])

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        # Optional custom projection layer: DenseNet -> Custom Projection -> ViT embedding
        if self.use_custom_proj:
            self.custom_proj = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(512, self.vit.embed_dim, kernel_size=1)
            )
        else:
            self.direct_projection = nn.Conv2d(1024, self.vit.embed_dim, kernel_size=1)

        # New classification head
        self.classifier = nn.Linear(self.vit.embed_dim, num_classes)

    def forward(self, images):
        batch_size = images.size(0)
        densenet_features = self.backbone(images)[-1]

        if self.use_custom_proj:
            projected_features = self.custom_proj(densenet_features)
        else:
            projected_features = self.direct_projection(densenet_features)

        patch_embeddings = projected_features.flatten(2).transpose(1, 2)
        cls_tokens_expanded = self.vit.cls_token.expand(batch_size, -1, -1)
        vit_input = torch.cat((cls_tokens_expanded, patch_embeddings), dim=1)
        vit_input = vit_input + self.vit.pos_embed[:, :vit_input.size(1), :]
        vit_input = self.vit.pos_drop(vit_input)

        vit_output = self.vit.blocks(vit_input)
        vit_output = self.vit.norm(vit_output)

        logits = self.classifier(vit_output[:, 0])
        return logits


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    img_dir = "images"
    csv_path = "Data_Entry_2017.csv"
    class_names = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
        "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
        "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax"
    ]

    # Domain aware transformations from the HyCoViT paper
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(csv_path)
    df = df[df["Image Index"].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))]
    df["Finding Labels"] = df["Finding Labels"].astype(str)

    unique_patient_ids = df['Patient ID'].unique()
    train_val_patient_ids, test_patient_ids = train_test_split(unique_patient_ids, test_size=0.2, random_state=42)
    train_patient_ids, val_patient_ids = train_test_split(train_val_patient_ids, test_size=0.25, random_state=42)

    train_df = df[df['Patient ID'].isin(train_patient_ids)]
    val_df = df[df['Patient ID'].isin(val_patient_ids)]
    test_df = df[df['Patient ID'].isin(test_patient_ids)]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_dataset = ChestXrayDataset(train_df, img_dir, class_names, transform=transform_train)
    val_dataset = ChestXrayDataset(val_df, img_dir, class_names, transform=transform_test)
    test_dataset = ChestXrayDataset(test_df, img_dir, class_names, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    model = DenseNetViT(use_custom_proj=False, freeze_backbone=True, freeze_vit=True, num_classes=len(class_names)).to(device)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    best_thresholds = None

    for epoch in range(1, 4):
        model.train()
        total_train_loss = 0
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(f"\nEpoch {epoch} | Train Loss: {total_train_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        all_val_labels, all_val_probs = [], []
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                batch_probs = torch.sigmoid(model(batch_images))
                all_val_labels.append(batch_labels.cpu())
                all_val_probs.append(batch_probs.cpu())

        all_val_labels = torch.cat(all_val_labels).numpy()
        all_val_probs = torch.cat(all_val_probs).numpy()

        # Compute per class thresholds and F1
        best_thresholds = get_optimal_thresholds(all_val_labels, all_val_probs)
        preds_binary = np.zeros_like(all_val_probs)
        for i in range(len(class_names)):
            preds_binary[:, i] = (all_val_probs[:, i] > best_thresholds[i]).astype(int)

        f1_scores = [f1_score(all_val_labels[:, i], preds_binary[:, i]) for i in range(len(class_names))]
        print("\nValidation F1 Scores:")
        for cls_name, f1_val, threshold in zip(class_names, f1_scores, best_thresholds):
            print(f"  {cls_name}: F1={f1_val:.3f} | Thr={threshold:.3f}")
        print(f"Macro F1: {np.mean(f1_scores):.3f}")

    # Test
    model.eval()
    all_test_labels, all_test_probs = [], []
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            batch_probs = torch.sigmoid(model(batch_images))
            all_test_labels.append(batch_labels.cpu())
            all_test_probs.append(batch_probs.cpu())

    all_test_labels = torch.cat(all_test_labels).numpy()
    all_test_probs = torch.cat(all_test_probs).numpy()

    preds_binary = np.zeros_like(all_test_probs)
    for i in range(len(class_names)):
        preds_binary[:, i] = (all_test_probs[:, i] > best_thresholds[i]).astype(int)

    f1_scores = [f1_score(all_test_labels[:, i], preds_binary[:, i]) for i in range(len(class_names))]
    auc_scores = [roc_auc_score(all_test_labels[:, i], all_test_probs[:, i]) for i in range(len(class_names))]

    print("\nFinal Test Results:")
    for cls_name, f1_val, auc_val in zip(class_names, f1_scores, auc_scores):
        print(f"  {cls_name}: F1={f1_val:.3f} | AUC={auc_val:.3f}")
    print(f"Test Macro F1: {np.mean(f1_scores):.3f}, Avg AUC: {np.mean(auc_scores):.3f}")


if __name__ == "__main__":
    main()

