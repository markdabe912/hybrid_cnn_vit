import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import torch.distributed as dist
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image


# =====================================================
# DEVICE SETUP
# =====================================================
# def get_device():
#     """Selects CUDA / MPS / CPU properly."""
#     if torch.cuda.is_available():
#         print(f"Using CUDA ({torch.cuda.device_count()} GPUs)")
#         return torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         print("Using Apple MPS")
#         return torch.device("mps")
#     else:
#         print("Using CPU")
#         return torch.device("cpu")

def get_device(requested_device="auto"):
    """
    Returns:
        device: torch.device(...)
        ddp_active: True/False
        rank: global rank
        world_size: total number of processes
        local_rank: local GPU index for this process
    """

    # -----------------------------------------
    # 1. Detect torchrun (DDP mode)
    # -----------------------------------------
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Read distributed environment variables
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # Set the correct GPU for this process
        torch.cuda.set_device(local_rank)

        # Initialize process group ONLY once
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}")

        return device, True, rank, world_size, local_rank


    # -----------------------------------------
    # 2. Single-GPU or CPU mode
    # -----------------------------------------
    req = requested_device.lower()

    # Manual override
    if req == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), False, 0, 1, 0
    if req == "mps" and torch.backends.mps.is_available():
        return torch.device("mps"), False, 0, 1, 0
    if req == "cpu":
        return torch.device("cpu"), False, 0, 1, 0

    # Auto detection
    if torch.cuda.is_available():
        return torch.device("cuda"), False, 0, 1, 0
    if torch.backends.mps.is_available():
        return torch.device("mps"), False, 0, 1, 0

    return torch.device("cpu"), False, 0, 1, 0


# =====================================================
# LOAD IMAGE FOLDERS
# =====================================================
def load_image_folders(img_dir):
    """
    Builds a dictionary:
        {'00000001.png': '/path/to/images_001/images', ...}
    """
    mapping = {}

    for i in range(1, 13):
        folder = os.path.join(
            img_dir, f"images_{str(i).zfill(3)}", "images"
        )

        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(".png"):
                    mapping[f] = folder

    return mapping

# =====================================================
# AUTO THRESHOLD
# =====================================================

# Index Issue
# precision: N
# recall: N
# thresholds: N-1
# np.argmax(f1_score) indexes into an arrau of length N, but thresh has length N-1 (mismatch)

def get_optimal_thresholds(labels, preds):
    thresholds = []
    for i in range(preds.shape[1]):
        precision, recall, thresh = precision_recall_curve(labels[:, i], preds[:, i])
        # Question:
        # f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
        best_thresh = thresh[np.argmax(f1_scores)] if len(thresh) > 0 else 0.5
        thresholds.append(best_thresh)
    return thresholds


# =====================================================
# MULTI-LABEL ACCURACY
# =====================================================
def multilabel_accuracy(y_true, y_pred):
    """
    Computes multi-label accuracy: correct labels / total labels.
    """
    correct = (y_true == y_pred).sum()
    total = y_true.size
    return correct / total
    


#### later use for plot

def plot_curves(train_loss_history, train_acc_history,
                valid_loss_history, valid_acc_history,
                save_dir_fig):

    # Ensure directory exists
    os.makedirs(save_dir_fig, exist_ok=True)

    # ----------------------------
    # 1. LOSS CURVE
    # ----------------------------
    fig1, ax1 = plt.subplots()
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.plot(train_loss_history, color='blue', label='Train Loss')
    ax1.plot(valid_loss_history, color='red', label='Val Loss')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_locator(MultipleLocator(1))   # better than 0.5
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir_fig, "loss_plot.png"))

    # ----------------------------
    # 2. ACCURACY CURVE
    # ----------------------------
    fig2, ax2 = plt.subplots()
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.plot(train_acc_history, color='blue', label='Train Acc')
    ax2.plot(valid_acc_history, color='red', label='Val Acc')
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir_fig, "accuracy_plot.png"))

    plt.show()




#confusion matrix function
def plot_cm(model, dl, save_dir_fig, device,normalize='true', test = False, report = False):
    #plot the confusion matrix
    categories = class_names
    model.eval()
    y_pred = []
    y_true = []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        output = model(x) #out shape: (batch_size, 5)
        y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
        y_true.extend(y.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    sns.heatmap(cm, annot=True, fmt= '.2f', cmap='Blues', xticklabels=categories.values(), yticklabels=categories.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if test:
        plt.savefig(os.path.join(save_dir_fig, "confusion_matrix_test.png"))
    else:
        plt.savefig(os.path.join(save_dir_fig, "confusion_matrix_val.png"))
    plt.show()
    if report:
        print(classification_report(y_true, y_pred, target_names=class_names.values()))



def vit_attn_map(attn_matx, batch_idx = 0):
    # Select which batch item and the first transformer block attention to visualize
    print("Num blocks:", len(attn_matx))
    print("Shape block 11:", attn_matx[0].shape)
    print("batch_idx:", batch_idx)
    attn = attn_matx[0][batch_idx]
    
    
    # Average across all heads
    attn = attn.mean(0)
    num_patches = attn_matx[0].shape[-1]-1
    grid_patch = int(np.sqrt(num_patches))
    # convert CLS token to patch attentions
    cls_attn = attn[0,1:]

    #detach from cpu
    cls_attn = cls_attn.detach().cpu().numpy()

    # compute patch grid size dynamically
    #patch_num = cls_attn.shape[1]
    #side = int(patch_num ** 0.5)

    #Reshape to (H,W) on cuda
    #attnmap_reshape = cls_attn.reshape(14,14)
    attnmap_reshape = cls_attn.reshape(grid_patch,grid_patch)
    #Upscale to original image size
    attnmap_ups = cv2.resize(attnmap_reshape, (224,224))

    #Normalize
    attnmap_ups -= attnmap_ups.min()
    attnmap_ups /= (attnmap_ups.max() + 1e-8)

    return attnmap_ups



def plot_attn_map(img,img_name,attnmap, save_path = None):

    orig_img = np.array(img.resize((224,224)))
    attn_heat = (attnmap * 255).astype(np.uint8)
    attn_heat = cv2.applyColorMap(attn_heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_img, 0.6, attn_heat, 0.4, 0)

    fig, ax = plt.subplots(1,2, figsize = (10,10))
    plt.axis("off")
    fig.suptitle(f"{img_name} VIT Attention Map")
    ax[0].imshow(orig_img)
    ax[0].set_title("Original X-ray")
    ax[0].set_axis_off()

    ax[1].imshow(overlay)
    ax[1].set_title(f"Attention Map Overlay")
    ax[1].set_axis_off()
    plt.axis("off")
    #plt.figure(figsize =(8,8))
    #plt.imshow(overlay)
    #plt.axis("off")
    plt.show()
    if save_path:
        fig.savefig(save_path,bbox_inches = "tight")
    plt.close(fig)

# =====================================================
# CREATE IMAGE FOLDERS DICTIONARY FOR COMBINED DATASET
# =====================================================
def create_dataset_dict(df_path, key_col="Image Index", val_col="path"):
    mapping = {}
    df = pd.read_csv(df_path)
    for _, row in df.iterrows():
        key = row[key_col]
        path_full = row[val_col]
        folder_pth = os.path.dirname(path_full)
        if row["source"] == "NIH":
            mapping[key] = "../" + folder_pth
        else:
            mapping[key] = "../" + path_full

    return mapping
