# Chest X-ray Multi-Label Classification — 9-Scripts Modular Framework

This project implements a **modular deep learning system** for multi-label Chest X-ray disease classification using a **DenseNet121 + Vision Transformer (ViT)** hybrid architecture.

The system is structured into **nine separate** scripts, each dedicated to a distinct stage of the workflow—including data loading and combination, preprocessing, model definition, training, evaluation, visualization, and features generation. This modular pipeline ensures clean organization, maintainability, and scalability, making it well-suited for experimentation and future extension.

In developing this system, portions of the implementation were adapted from the “Reproducing and Improving CheXNet” repository. The original work by dstrick17 is provided under the MIT License, and the relevant components have been incorporated responsibly and with attribution.

---

Before running any part of this project, make sure your environment is properly prepared. Follow the steps below to set up the dataset, install dependencies, and ensure everything is ready for execution.

1) Download the NIH images -- download_dataset.sh

2) Download the Chexpert images -- download_dataset2.sh

3) Install Dependencies -- requirements.txt

4) Install opencv and Pillow -- Please install opencv-python(to import cv2) and Pillow versions compatible with numpy 1.26.4. after running the requirements.txt. Note these are not included in the requirement.txt file because of installation issues encountered. Please install separately before running any scripts and be aware of opencv silently upgrading numpy from 1.26.4 to numpy 2+. if you install a newer version of opencv.

---

## 1. `dataset.py` — Dataset & Label Encoding

### 1.1 ChestXrayDataset
- Loads X-ray images from folders
- Applies preprocessing transforms
- Converts “Finding Labels” (multi-label text) into a **14-dimensional binary vector**

### 1.2 get_label_vector()
- Converts label strings into model-ready label vectors

### 1.3 location 
- "./data/dataset.py

**Purpose:**  
Provide clean, preprocessed images and labels for training.

---

## 2. `CS7643projectcomb-final.ipynb` — Combined dataset Jupyter notebook  

### 2.1 
- Assumes both datasets have been downloaded into folders created by the download_dataset and download_dataset2 scripts
- datasets folder should be in the same directory as this jupyter notebook
- script combines the NIH and Chexpert datasets into a unified dataframe and saves the dataframe as a csv file **nih_chexcomb_df_final.csv** to your local directory 
- generates and saves disease prevalence plots **NIH Imbalance Plot** and **NIH_Chex Imbalance Plot** to your local directory
### 2.2 location
- "./data/combine_dataset.py"??

**Purpose:**  
Create a unified combined dataset csv file for training. Generate disease prevalence plots for report. User needs to update csv_path argument in config_HCV.yaml file to **nih_chexcomb_df_final.csv** file path if training combined dataset

---

## 3.`losses.py` — Focal Loss for Imbalanced Data

### 3.1 FocalLoss
A loss function effective for:
- Highly imbalanced medical datasets
- Rare disease classes

### 3.2 location
- "./scripts/losses.py"

**Purpose:**  
Improve detection of underrepresented diseases.

---

## 4. `HCV_model.py` — DenseNet + ViT Hybrid Model

### 4.1 DenseNetViT
A custom architecture combining:
- **DenseNet121** for local feature extraction  
- **Vision Transformer (Vit)** for global pattern reasoning

### 4.2 Features:
- Freeze/unfreeze backbone and ViT
- Optional custom projection (DenseNet → ViT embed dim)
- Outputs logits for 14 diseases
- **version in the combined dataset folder contains code to register hook to capture the attention weights during the forward pass**

### 4.3 location
- "./scripts/HCV_model.py"
- "./scripts/HCV_model.py" **need to update this location for version in combined dataset folder**

**Purpose:**  
Fuse CNN local features with Transformer global context for stronger medical image performance.
**Combined dataset version: In addition, register hook to capture attention weights during the forward pass**

---

## 5. `utilis.py` — Utility Functions


### 5.1 load_image_folders()  
- Maps each image file to its containing folder. Used for NIH dataset only

### 5.2 get_device() 
- Automatically selects GPU → MPS → CPU.

### 5.3 get_optimal_thresholds() 
- Computes the **best F1 threshold per disease** using precision-recall curves.

### 5.4 multilabel_accuracy() 
- Computes multi-label accuracy cleanly.
**Added by Titi**
### 5.5 vtt_attn_map() 
- Create the attention map for visualization.

### 5.6 plot_attn_map() 
- Plots a pair of images, the original image and an overlay of the created attention map on the original map and save plot to your local directory.

### 5.7 create_dataset_dict() 
- create image_to_folder dictionary for combined dataset

### 5.8 location
- "./scripts/utilis.py"

**Purpose:**  
Provide reusable utility functions for the trainer.

---

## 6. `trainer.py` — Full Training, Validation & Testing Pipeline

### This is the core training engine (ChestXrayHander).

### 6.1 Training Loop  
- Computes loss, accuracy  
- Updates model weights
- **version in combined dataset fire the hook to capture attention weight** 

### 6.2 Validation Loop  
- Computes macro F1, optimal thresholds, loss, accuracy 
- Tracks best model  

### 6.3 Test Evaluation  
- Outputs final F1, AUC, accuracy

### 6.4 Checkpoints  
Saves:
- best_model.pth
- last_checkpoint.pth

### 6.5 Resume Training  
Reloads:
- Model weights  
- Optimizer & scheduler  
- Thresholds  
- Epoch index

### 6.6 generate_vit_map() 
- Preprocess an input image
- Run inference to capture attention weights
- Run vit_attn_map to create the image's attention map
- Run plot_attn_map to generate and save visualization

### 6.7 location
- "./scripts/trainer.py"
- "./scripts/trainer.py" **need to update location for version in combined dataset**

**Purpose:**  
Run the entire machine learning workflow end-to-end.

---

## 7 `main.py` — Command-Line Entry Point

### This script is the interface for users:

### 7.1 Parses arguments  
(e.g., epochs, LR, freezing backbone, paths from yaml file)

### 7.2 Instantiates ChestXrayTrainer

### 7.3 Runs the full training and test pipeline  
via trainer.run()

### 7.4 location
- "./scripts/main.py"

**Purpose:**  
Simple, clean CLI to run experiments.



## 8 Grad-CAM script



**Purpose:**  

This script generates Grad-CAM images for the 4 dense blocks of a DenseNet backbone CNN. 
It visualizes the areas of an image that contributed the most to the model's prediction. 
Also, the script  combines these individual Grad-CAM images into 1 final image with each heatmap in a separate quadrant.

### 1. **Prepare the Files:**
   - Place the script in the same directory as the model definition `HCV_model.py` and the pre-trained weights file (for example, `best_checkpoint.pth`).
   - Make sure the image you want to process (for example, `00000001_001.png`) is in the same directory.
   - Set the image filename in the script by updating the `image_path` variable:

     ```python
     image_path = "image_name.png"
     ```

### 2. **Run the Script:**
   - Execute the script by running:

       ```bash
       python gradcam.py
       ```




---

## 9 `generate_attnmaps.py` —
**Purpose:**  
This script loads a model from a checkpoint, default is the best checkpoint and runs the generate_vit_map function in trainer.py to generate and save attention map plots for an input image. 

### 1. **Run the Script:**
   - requires installation of opencv and Pillow versions compatible with numpy 1.26.4. Note these are not included in the requirement file because of installation issues encountered. Please install separately before running and be aware of opencv silently upgrading numpy from 1.26.4 to numpy 2+.
   - update checkpoint path in the file if you want to use another model different from the best model
   - update test_image with the file name of the image you want to generate attention map for e.g "00000008_002.png"
   - update path to where to save the generated attention map plot. Default is the results folder that is on the same        level as the scripts folder

### 9.x location
- "./visualizations/generate_attnmaps.py" **need to update after cleanup**

### This script is the interface for users:

### 9.1 xxx



### 9.x location
- "./visualizations/.py"

**Purpose:**  

---


## 10. Summary Table

| Scripts        | Functionality                           |
|-----------------|-------------------------------------------|
| `dataset.py`    | Dataset loading & label encoding          |
| `CS7643projectcomb-final.ipynb`      |            |
| `losses.py`       | FocalLoss implementation                 |
| `HCV_model.py`      | DenseNet + ViT hybrid model              |
| `utilis.py`      | Utilities: thresholds, accuracy, device  |
| `trainer.py`    | Training/validation/testing pipeline     |
| `main.py`      | CLI entry point for training             |
| `gradcam.py`      | Generates Grad-CAM images for the 4 dense blocks of a DenseNet backbone CNN   | 
| `attention.py`      |       xxx       |



