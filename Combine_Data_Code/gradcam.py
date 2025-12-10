import torch
import cv2
import numpy as np
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from HCV_model import DenseNetViT

model = DenseNetViT(use_custom_proj=True,  custom_proj_layers=256)
ckpt = torch.load("best_checkpoint.pth", map_location="cpu")
state_dict = ckpt["model_state"]
model.load_state_dict(state_dict, strict=False)

device = torch.device("cpu")
model.to(device)
model.eval()


def preprocess_image(image_path, resize=(224, 224)):
    original_img = cv2.imread(image_path)[:, :, ::-1]
    img_resized = cv2.resize(original_img, resize)
    img_normalized = img_resized.astype(np.float32) / 255.0
    input_tensor = torch.tensor(img_normalized.transpose(2, 0, 1)).unsqueeze(0)
    return input_tensor.to(device), original_img

# image_path = "view1_frontal.jpg"
image_path = "00000001_001.png"
input_tensor, original_img = preprocess_image(image_path)


with torch.no_grad():
    output = model(input_tensor)



disease_list = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]



probabilities = torch.softmax(output, dim=1)
probs_np = probabilities.cpu().numpy()[0]

print("Softmax probabilities for all classes:\n")
for i, p in enumerate(probs_np):
    print(f"{i:02d}  {disease_list[i]:20s} : {p:.6f}")

predicted_label = probs_np.argmax()
print("\nPredicted class:", disease_list[predicted_label], f"({predicted_label})")

dense_blocks = {
    "denseblock1": model.backbone.features_denseblock1,
    "denseblock2": model.backbone.features_denseblock2,
    "denseblock3": model.backbone.features_denseblock3,
    "denseblock4": model.backbone.features_denseblock4,
}

os.makedirs("cams_all_blocks", exist_ok=True)
saved_paths = []

for block_name, layer in dense_blocks.items():
    print(f"Generating CAM for {block_name}...")

    cam = GradCAM(model=model, target_layers=[layer])
    pred_target = [ClassifierOutputTarget(predicted_label)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=pred_target)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (original_img.shape[1], original_img.shape[0]))

    cam_image = show_cam_on_image(original_img.astype(np.float32) / 255.0,
                                  grayscale_cam, use_rgb=True)

    out_path = f"cams_all_blocks/cam_{block_name}_pred.jpg"
    cv2.imwrite(out_path, cam_image[:, :, ::-1])
    saved_paths.append(out_path)

    print(f"Saved: {out_path}")


imgs = [cv2.imread(p) for p in saved_paths]

h, w = imgs[0].shape[:2]
imgs = [cv2.resize(img, (w, h)) for img in imgs]

top_row = np.hstack((imgs[0], imgs[1]))
bottom_row = np.hstack((imgs[2], imgs[3]))

quadrant = np.vstack((top_row, bottom_row))

final_path = f"cams_all_blocks/{image_path}cam_quadrants.jpg"
cv2.imwrite(final_path, quadrant)
