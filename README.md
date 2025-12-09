# Chest X-ray Multi-Label Classification — 9-Script Modular Framework

This project implements a **modular deep learning system** for multi-label Chest X-ray disease classification using a **DenseNet121 + Vision Transformer (ViT)** hybrid architecture.

The system is structured into **nine separate** scripts, each dedicated to a distinct stage of the workflow—including data loading and combination, preprocessing, model definition, training, evaluation, visualization, and features generation. This modular pipeline ensures clean organization, maintainability, and scalability, making it well-suited for experimentation and future extension.

In developing this system, portions of the implementation were adapted from the “Reproducing and Improving CheXNet” repository. The original work by dstrick17 is provided under the MIT License, and the relevant components have been incorporated responsibly and with attribution.

---

Download the image -- download_dataset.sh

Install Dependencies -- requirements.txt

---

## 1. `dataset.py` — Dataset & Label Encoding

### 1.1 ChestXrayDataset
- Loads X-ray images from folders
- Applies preprocessing transforms
- Converts “Finding Labels” (multi-label text) into a **14-dimensional binary vector**

### 1.2 get_label_vector()
- Converts label strings into model-ready label vectors

**Purpose:**  
Provide clean, preprocessed images and labels for training.

---

## 2.`losses.py` — Focal Loss for Imbalanced Data

### 2.1 FocalLoss
A loss function effective for:
- Highly imbalanced medical datasets
- Rare disease classes

**Purpose:**  
Improve detection of underrepresented diseases.

---

## 3. `HCV_model.py` — DenseNet + ViT Hybrid Model

### 3.1 DenseNetViT
A custom architecture combining:
- **DenseNet121** for local feature extraction  
- **Vision Transformer (Vit)** for global pattern reasoning  

Features:
- Freeze/unfreeze backbone and ViT
- Optional custom projection (DenseNet → ViT embed dim)
- Outputs logits for 14 diseases

**Purpose:**  
Fuse CNN local features with Transformer global context for stronger medical image performance.

---

## 4. `utilis.py` — Utility Functions


### 4.1 load_image_folders()  
Maps each image file to its containing folder.

### 4.2 get_device() 
Automatically selects GPU → MPS → CPU.

### 4.3 get_optimal_thresholds() 
Computes the **best F1 threshold per disease** using precision-recall curves.

### 4.4 multilabel_accuracy() 
Computes multi-label accuracy cleanly.

### 4.5 plot()
Plot the training, validation and testing results

**Purpose:**  
Provide reusable utility functions for the trainer.

---

## 5. `trainer.py` — Full Training, Validation & Testing Pipeline

### This is the core training engine.

### 5.1 Training Loop  
- Computes loss, accuracy  
- Updates model weights  

### 5.2 Validation Loop  
- Computes macro F1, optimal thresholds, loss, accuracy 
- Tracks best model  

### 5.3 Test Evaluation  
- Outputs final F1, AUC, accuracy

### 5.4 Checkpoints  
Saves:
- best_model.pth
- last_checkpoint.pth

### 5.5 Resume Training  
Reloads:
- Model weights  
- Optimizer & scheduler  
- Thresholds  
- Epoch index  

**Purpose:**  
Run the entire machine learning workflow end-to-end.

---

## 6 `main.py` — Command-Line Entry Point

### This script is the interface for users:

### 6.1 Parses arguments  
(e.g., epochs, LR, freezing backbone, paths from yaml file)

### 6.2 Instantiates ChestXrayTrainer

### 6.3 Runs the full training pipeline  
via trainer.run()

**Purpose:**  
Simple, clean CLI to run experiments.

---

## 7. Summary Table

| Scripts        | Functionality                           |
|-----------------|-------------------------------------------|
| `dataset.py`    | Dataset loading & label encoding          |
| `losses.py`       | FocalLoss implementation                 |
| `HCV_model.py`      | DenseNet + ViT hybrid model              |
| `utilis.py`      | Utilities: thresholds, accuracy, device, plot  |
| `trainer.py`    | Training/validation/testing pipeline     |
| `main.py`      | CLI entry point for training             |

