import torch
import cv2
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from hybrid_cnn_vit import DenseNetViT
model = DenseNetViT()

checkpoint_path = "best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)

device = torch.device("cpu")
model.to(device)
model.eval()

def preprocess_image(image_path, resize=(224, 224)):
    original_img = cv2.imread(image_path)[:, :, ::-1]  # BGR → RGB
    img_resized = cv2.resize(original_img, resize)
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = torch.tensor(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    return input_tensor.to(device), original_img

true_label = 3

image_path = "00000001_001.png"
input_tensor, original_img = preprocess_image(image_path)

target_layer = model.backbone.features_denseblock3
cam = GradCAM(model=model, target_layers=[target_layer])

with torch.no_grad():
    output = model(input_tensor)
predicted_label = output.argmax(dim=1).item()

pred_target = [ClassifierOutputTarget(predicted_label)]
pred_cam = cam(input_tensor=input_tensor, targets=pred_target)[0]
pred_cam = cv2.resize(pred_cam, (original_img.shape[1], original_img.shape[0]))

pred_cam_image = show_cam_on_image(original_img.astype(np.float32)/255.0,  pred_cam, use_rgb=True)

cv2.imwrite("cam_predicted.jpg", pred_cam_image[:, :, ::-1])

true_target = [ClassifierOutputTarget(true_label)]
true_cam = cam(input_tensor=input_tensor, targets=true_target)[0]
true_cam = cv2.resize(true_cam, (original_img.shape[1], original_img.shape[0]))

true_cam_image = show_cam_on_image(original_img.astype(np.float32)/255.0,
                                   true_cam, use_rgb=True)

cv2.imwrite("cam_true.jpg", true_cam_image[:, :, ::-1])
