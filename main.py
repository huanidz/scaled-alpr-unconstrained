import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(11)
from components.model.AlprModel import AlprModel
from time import perf_counter
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 normalization
])

# trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
# testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

model = AlprModel().to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')
# print(f"==>> model: {model}")

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# # Training loop
# for epoch in range(10):  # Loop over the dataset multiple times
#     for i, data in enumerate(trainloader, 0):
#         # Get the inputs and move them to the specified device
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

# print('Finished Training')


# print("Start warming up 100 iteration batch=1..")
start_time = perf_counter()
input = torch.randn((32, 3, 480, 480), dtype=torch.float32).to(device)
probs, bbox = model(input)
print(f"==>> probs.shape: {probs.shape}")
print(f"==>> bbox.shape: {bbox.shape}")
probs = torch.softmax(probs)
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print('Interrupted by user')
    

    
end_time = perf_counter()
elaps = end_time - start_time
print(f"Finish warming up. Took {elaps}ms total. Average = {elaps}ms per iteration.")
