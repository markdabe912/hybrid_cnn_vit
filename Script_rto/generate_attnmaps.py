from trainer import ChestXrayHandler
from HCV_model import DenseNetViT
import torch
from PIL import Image
import os

if __name__ == "__main__" :

    trainer = ChestXrayHandler(args)
    trainer.build_transforms()
    trainer.build_model()
    trainer.load_checkpoint("../checkpoints/best_checkpoint.pth")

    test_image = "00000008_002.png"
    trainer.generate_vit_map(batch= None,img_name = test_image, batch_index = 0, save_dir = "../results")