import torch
from torchvision import models, transforms
from time import perf_counter
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#Set the pretrained parameter as True when loading the model to get the pretrained weights.
model = models.resnet50(pretrained=True).to(device)
print(f"==>> model: {model}")
print(f'The model has {count_parameters(model):,} trainable parameters')

print("Start warming up 100 iteration batch=1..")
start_time = perf_counter()
input = torch.randn((12, 3, 256, 256), dtype=torch.float32).to(device)
output = model(input)
print(f"==>> output.shape: {output.shape}")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Interrupted by user')
    

    
end_time = perf_counter()
elaps = end_time - start_time
print(f"Finish warming up. Took {elaps}ms total. Average = {elaps}ms per iteration.")

