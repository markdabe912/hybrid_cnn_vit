

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


# "Borrowed" from https://github.com/dstrick17/DacNet/blob/main/scripts/dacnet.py

# =====================================================
# Label encoder
# =====================================================
def get_label_vector(labels_str, disease_list):
    labels = labels_str.split('|')
    if labels == ['No Finding']:
        return [0] * len(disease_list)
    return [1 if disease in labels else 0 for disease in disease_list]


# =====================================================
# Dataset
# =====================================================
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, image_to_folder, disease_list, transform=None):
        self.dataframe = dataframe
        self.image_to_folder = image_to_folder
        self.transform = transform
        self.disease_list = disease_list

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        folder = self.image_to_folder[img_name]

        img_path = os.path.join(folder, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_vector = get_label_vector(
            self.dataframe.iloc[idx]['Finding Labels'],
            self.disease_list
        )
        labels = torch.tensor(label_vector, dtype=torch.float)

        return image, labels
