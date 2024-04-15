import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from natsort import natsorted

class AlprDataset(Dataset):
    
    def __init__(self, data_folder) -> None:
        super(AlprDataset, self).__init__()
        
        total_items = glob(f"{data_folder}/*")

    