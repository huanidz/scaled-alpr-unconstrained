import torch
torch.manual_seed(11)
from components.model.AlprModel import AlprModel
from components.losses.loss import AlprLoss
from time import perf_counter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.util_func import count_parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
])

model = AlprModel().to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')
loss = AlprLoss().to(device)













# print("Start warming up 100 iteration batch=1..")
start_time = perf_counter()
input = torch.randn((16, 3, 384, 384), dtype=torch.float32).to(device)
probs, bbox = model(input)

# Simulate the loss backprop
target_probs = torch.randn((16, 2, 24, 24), dtype=torch.float32).to(device)
target_bbox = torch.randn((16, 6, 24, 24), dtype=torch.float32).to(device)

dummy_target_output = torch.randn((16, 9, 24, 24), dtype=torch.float32).to(device)
concat_predict_output = torch.cat([probs, bbox], dim=1)
loss = AlprLoss()
loss_val = loss(concat_predict_output, dummy_target_output)
loss_val.backward()
end_time = perf_counter()
elaps = end_time - start_time
print(f"Finish warming up. Took {elaps}ms total. Average = {elaps}ms per iteration.")
