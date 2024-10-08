from components.model.AlprModel import AlprModel
import torch
from utils.util_func import count_parameters

rand_in = torch.rand(1, 3, 384, 384)

model = AlprModel(scale="small", feature_extractor="dla")

print("Total parameters: ", count_parameters(model))

probs, bbox = model(rand_in)
print(f"==>> probs.shape: {probs.shape}")
print(f"==>> bbox.shape: {bbox.shape}")