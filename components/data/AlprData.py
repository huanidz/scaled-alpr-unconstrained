import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset

class AlprDataset(Dataset):
    
    def __init__(self, images_folder, labels_folder, input_size=384) -> None:
        super(AlprDataset, self).__init__()
        
        self.images = natsorted(glob(f"{images_folder}/*"))
        self.labels = natsorted(glob(f"{labels_folder}/*"))
        self.input_size = input_size
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.load_image(self.images[index])
        label = self.load_label(self.labels[index])
        return image, label
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        return image
    
    def load_label(self, label_path):
        label = np.genfromtxt(label_path, delimiter=",")
        label = torch.from_numpy(label).float()
        return label    