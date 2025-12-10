import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
import torch.distributed as dist

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
    







    
