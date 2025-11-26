import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from dataset import ChestXrayDataset
from HCV_model import DenseNetViT
from losses import FocalLoss
from utilis import (
    get_device,
    load_image_folders,
    get_optimal_thresholds,
    multilabel_accuracy,
)
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

    
# =====================================================
# TRAINER CLASS (MAIN PART)
# =====================================================
class ChestXrayHandler:
    def __init__(self, args):
        self.args = args
        
        # Device + Distributed 
        self.device, self.ddp_active, self.rank, self.world_size, self.local_rank = get_device(self.args.device)
        self.is_master = (self.rank == 0)

        if self.is_master:
            print(f"Device: {self.device}, Rank: {self.rank}, DDP: {self.ddp_active}")

        # Disease Classes 
        self.disease_list = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Effusion", "Emphysema", "Fibrosis", "Hernia",
            "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
            "Pneumonia", "Pneumothorax"
        ]
        
        #  Prepare Save_dir
        os.makedirs(args.save_dir, exist_ok=True)

        # Load Images 
        self.img_map = load_image_folders(args.img_dir)
        
        # Split Data
        self.train_df, self.val_df, self.test_df = self.split_data(args.csv_path)
        
        # Build transforms & loaders
        self.train_tf, self.test_tf = self.build_transforms()
        self.train_loader, self.val_loader, self.test_loader = self.build_loaders()

        # Build model
        self.model, self.optimizer, self.criterion, self.lr_scheduler = self.build_model()

        # Track best macro-F1
        self.best_macro_f1 = -1 
        self.best_thresholds = None
        self.start_epoch = 1

        # Logs
        self.history = {
            "epoch": [],
            "train_loss": [], 
            "train_acc": [], 
            "val_loss": [], 
            "val_acc": [], 
            "val_macro_f1": []
        }
     
        # Prepare checkpoint directory
        # Always keep checkpoints in the project root
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Determine resume path
        # Case 1: user explicitly disables resume

        if args.test_model and args.resume in ["none", "", "None", None, "auto"]:
            best_resume = os.path.join(self.ckpt_dir, "best_checkpoint.pth")
            args.resume = best_resume if os.path.exists(best_resume) else None
        elif args.resume == "auto":
            auto_resume = os.path.join(self.ckpt_dir, "last_checkpoint.pth")
            args.resume = auto_resume if os.path.exists(auto_resume) else None
        elif args.resume in ["none", "", "None"]:
            args.resume = None

        if args.resume is not None and os.path.exists(args.resume):
            self.load_checkpoint(args.resume)
            
        # ------------------------------------------------------

    def split_data(self, csv_path):
        df = pd.read_csv(csv_path)
        # Filter the CSV to include only images that are present in the folders
        df = df[df["Image Index"].isin(self.img_map.keys())]
        df["Finding Labels"] = df["Finding Labels"].astype(str)
        
        pids = df["Patient ID"].unique()
        train_val, test = train_test_split(pids, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.25, random_state=42)

        return (
            df[df["Patient ID"].isin(train)],
            df[df["Patient ID"].isin(val)],
            df[df["Patient ID"].isin(test)]
        )

    # ------------------------------------------------------

    def build_transforms(self):
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomAffine(5, translate=(0.02, 0.02)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
        test_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

        return train_tf, test_tf


    # ------------------------------------------------------

    def build_loaders(self):
        train_ds = ChestXrayDataset(self.train_df, self.img_map, self.disease_list, self.train_tf)
        val_ds   = ChestXrayDataset(self.val_df,   self.img_map, self.disease_list, self.test_tf)
        test_ds  = ChestXrayDataset(self.test_df,  self.img_map, self.disease_list, self.test_tf)

        # Distributed samplers for DDP
        train_sampler = DistributedSampler(train_ds) if self.ddp_active else None
        val_sampler   = DistributedSampler(val_ds, shuffle=False) if self.ddp_active else None
        test_sampler  = DistributedSampler(test_ds, shuffle=False) if self.ddp_active else None

        # return (
        #     DataLoader(train_ds, batch_size=self.args.batch_size, shuffle=True,num_workers=self.args.workers),
        #     DataLoader(val_ds,   batch_size=self.args.batch_size, shuffle=False,num_workers=self.args.workers),
        #     DataLoader(test_ds,  batch_size=self.args.batch_size, shuffle=False,num_workers=self.args.workers),
        # )
        
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=(not self.ddp_active),  # If ddp_active then no need to shuffle, otherwise shuffle
            sampler=train_sampler,
            num_workers=self.args.workers,
            drop_last=False ,
            persistent_workers=False,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.args.workers,
            drop_last=False ,
            persistent_workers=False,
        )

        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=self.args.workers,
            drop_last=False,
            persistent_workers=False,
        )

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        
        return self.train_loader, self.val_loader, self.test_loader
    # ------------------------------------------------------

    def build_model(self):
        # Build main model
        model = DenseNetViT(
            num_classes=len(self.disease_list),
            use_custom_proj=self.args.use_custom_proj,
            freeze_backbone=self.args.freeze_backbone,
            freeze_vit=self.args.freeze_vit,
            train_last_vit_layers=self.args.train_last_vit_layers,
            custom_proj_layers=self.args.custom_proj_layers
        ).to(self.device)

        # Wrap with DDP if needed
        if self.ddp_active:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
    
        # Store model
        self.model = model

        # Get inner model for accessing classifier etc.
        module = self.model.module if self.ddp_active else self.model
    
        # Optimizer
        if self.args.freeze_backbone and self.args.freeze_vit:
            self.optimizer = optim.Adam(module.classifier.parameters(), lr=self.args.lr)
        else:
            trainable = [p for p in module.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(trainable, lr=self.args.lr)
    
        # Loss function
        self.criterion = FocalLoss(alpha=1, gamma=2)
    
        # Scheduler
        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=self.args.lr_decay)
    
        # Return values
        return self.model, self.optimizer, self.criterion, self.lr_scheduler

    # ------------------------------------------------------
    # TRAINING EPOCH
    # ------------------------------------------------------

    def train_one_epoch(self, epoch):
        self.model.train()
        if self.train_sampler:
            self.train_sampler.set_epoch(epoch)
            
        total_loss = 0
        acc_correct = 0
        acc_total = 0

        #pbar = self.train_loader if not self.is_master else tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        if self.is_master:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} Training")
        else:
            pbar = self.train_loader

        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accuracy
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
                labs = labels.cpu().numpy()
                acc_correct += (preds == labs).sum()
                acc_total += preds.size

            if self.is_master:
                pbar.set_postfix(loss=loss.item())

        train_acc = acc_correct / acc_total
        avg_loss = total_loss / len(self.train_loader)
        
        if self.args.verbose and not self.is_master:
            print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        return train_acc, avg_loss


        
    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    def validate(self, epoch):
        self.model.eval()
        
        if self.val_sampler:
            self.val_sampler.set_epoch(epoch)
            
        all_labels, all_probs = [], []
        val_loss_total = 0
        acc_correct = 0
        acc_total = 0

        #loader = self.val_loader if not self.is_master else tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")


        if self.is_master:
            loader = tqdm(self.val_loader, desc=f"Epoch {epoch} Validation")
        else:
            loader = self.val_loader
            
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                logits = self.model(imgs)
                probs = torch.sigmoid(logits)
                loss = self.criterion(logits, labels)
                
                val_loss_total += loss.item()
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

                preds = (probs > 0.5).int().cpu().numpy()
                labs = labels.cpu().numpy()
                acc_correct += (preds == labs).sum()
                acc_total += preds.size
                
        if not self.is_master:
            return None

        val_acc = acc_correct / acc_total
        val_loss = val_loss_total / len(self.val_loader)

        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        thresholds = np.array(get_optimal_thresholds(all_labels, all_probs))

        preds_final = (all_probs > thresholds[None, :]).astype(int)
        # f1_scores = [f1_score(all_labels[:, i], preds_thr[:, i])
        #              for i in range(len(self.disease_list))]
        # macro_f1 = np.mean(f1_scores)

        f1_scores = []
        auc_scores = []

        for i in range(len(self.disease_list)):
            y_true = all_labels[:, i]
            y_prob = all_probs[:, i]
            y_pred = preds_final[:, i]

            if len(np.unique(y_true)) < 2:
                f1_scores.append(np.nan)
                auc_scores.append(np.nan)
                continue

            # Compute F1 safely
            try:
                f1 = f1_score(y_true, y_pred)
            except ValueError:
                f1 = np.nan
            f1_scores.append(f1)

            # Compute AUC safely
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = np.nan
            auc_scores.append(auc)
        
        macro_f1 = np.nanmean(f1_scores)

        # Save per-class metrics
        pd.DataFrame({
            "Disease": self.disease_list,
            "F1": f1_scores,
            "AUC": auc_scores
        }).to_csv(os.path.join(self.args.save_dir, "validation_per_class.csv"), index=False)

        if self.args.verbose:
            print(f"Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | Macro F1={macro_f1:.4f}")

        return macro_f1, thresholds, val_acc, val_loss

     
    # ------------------------------------------------------
    # TESTING
    # ------------------------------------------------------

    def test(self):
        print("\nRunning FINAL TEST evaluation...")
    
        self.model.eval()
        all_labels, all_probs = [], []
        acc_correct = 0
        acc_total = 0
        test_loss_total = 0
    
        with torch.no_grad():
            if self.is_master:
                loader = tqdm(self.test_loader, desc="Testing")
            else:
                loader = self.test_loader
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(imgs)
                probs = torch.sigmoid(logits)
    
                # accumulate loss
                loss = self.criterion(logits, labels)
                test_loss_total += loss.item()
    
                # collect preds + labels
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
    
                preds = (probs.cpu().numpy() > self.best_thresholds[None, :]).astype(int)
                labs = labels.cpu().numpy()
    
                acc_correct += (preds == labs).sum()
                acc_total += preds.size

        # final accuracy & loss
        test_acc = acc_correct / acc_total
        test_loss = test_loss_total / len(self.test_loader)
    
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
    
        thresholds = np.array(self.best_thresholds)
        preds_final = (all_probs > thresholds[None, :]).astype(int)
        
        f1_scores = []
        auc_scores = []
        
        for i in range(len(self.disease_list)):
            y_true = all_labels[:, i]
            y_prob = all_probs[:, i]
            y_pred = preds_final[:, i]
        
            if len(np.unique(y_true)) < 2:
                f1_scores.append(np.nan)
                auc_scores.append(np.nan)
                continue
        
            # Compute F1 safely
            try:
                f1 = f1_score(y_true, y_pred)
            except ValueError:
                f1 = np.nan
            f1_scores.append(f1)
        
            # Compute AUC safely
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = np.nan
            auc_scores.append(auc)

        macro_f1 = np.nanmean(f1_scores)
    
        # Print summary
        if self.args.verbose:
            print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Macro F1: {macro_f1:.4f}")


        # Save per-class metrics
        pd.DataFrame({
            "Disease": self.disease_list,
            "F1": f1_scores,
            "AUC": auc_scores
        }).to_csv(os.path.join(self.args.save_dir, "final_test_per_class.csv"), index=False)

        # Save global metrics
        pd.DataFrame({
            "metric": ["test_loss", "test_accuracy", "macro_f1"],
            "value": [test_loss, test_acc, np.mean(f1_scores)]
        }).to_csv(os.path.join(self.args.save_dir, "final_test_summary.csv"), index=False)
            
    # ------------------------------------------------------
    # load_checkpoint
    # ------------------------------------------------------

    def load_checkpoint(self, ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")

        # My local machine requires torch 2.8+ for cuda so this code is just to keep things compatible
        torch_version = torch.__version__.split('.')[:2]

        if int(torch_version[0]) > 2 or int(torch_version[0]) == 2 and int(torch_version[1]) >= 6:
            with torch.serialization.safe_globals([np._core.multiarray.scalar]):
                checkpoint = torch.load(ckpt_path, weights_only=False, map_location=self.device)
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
    
        # Restore model
        state_dict = checkpoint["model_state"]
        if self.ddp_active:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
    
        # -------- Safe Optimizer Load --------
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print("WARNING: Optimizer structure changed — skipping optimizer state load.")
            print(f"  Reason: {e}")
    
        # -------- Safe Scheduler Load --------
        try:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        except Exception as e:
            print("WARNING: Scheduler structure changed — skipping scheduler state load.")
            print(f"  Reason: {e}")
    
        # Restore metrics
        self.best_macro_f1 = checkpoint.get("best_macro_f1", -1)
        self.best_thresholds = checkpoint.get("best_thresholds", None)
    
        # Restore epoch
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}.")



    # ------------------------------------------------------
    # RUN FULL TRAINING PIPELINE
    # ------------------------------------------------------

    def run(self):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
    
            # ------------------ TRAIN ------------------
            train_acc, train_loss = self.train_one_epoch(epoch)
    
            if self.ddp_active:
                dist.barrier()
    
            if self.is_master:
                if self.args.verbose:
                    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    
                # ------------------ VALIDATION ------------------
                val_result = self.validate(epoch)
    
                if val_result is None:
                    raise RuntimeError("Validation failed to return results on master rank.")
    
                macro_f1, thresholds, val_acc, val_loss = val_result
    
                # Update LR
                self.lr_scheduler.step()
    
                # Save history
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_macro_f1"].append(macro_f1)
    
                pd.DataFrame(self.history).to_csv(
                    os.path.join(self.args.save_dir, "training_results.csv"),
                    index=False
                )
    
                # Save best model
                if macro_f1 > self.best_macro_f1:
                    self.best_macro_f1 = macro_f1
                    self.best_thresholds = thresholds
    
                    best_path = os.path.join(self.ckpt_dir, "best_checkpoint.pth")
                    torch.save({
                        "model_state": (
                            self.model.module.state_dict()
                            if self.ddp_active else self.model.state_dict()
                        ),
                        "best_thresholds": thresholds,
                        "best_macro_f1": macro_f1,
                        "epoch": epoch
                    }, best_path)
                    if self.args.verbose:
                        print(f"Saved BEST model → Macro F1: {macro_f1:.4f}")
    
                # Save last checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, "last_checkpoint.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": (
                        self.model.module.state_dict()
                        if self.ddp_active else self.model.state_dict()
                    ),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.lr_scheduler.state_dict(),
                    "best_macro_f1": self.best_macro_f1,
                    "best_thresholds": self.best_thresholds
                }, ckpt_path)
    
            if self.ddp_active:
                dist.barrier()
