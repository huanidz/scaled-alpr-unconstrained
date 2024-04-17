import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from components.model.AlprModel import AlprModel
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset 
from time import perf_counter
import numpy as np

loader = AlprDataset(images_folder="./train_data/images", labels_folder="./train_data/labels", input_size=384)
item = loader[0]
