import cv2
import torch
import numpy as np
from components.model.AlprModel import AlprModel
from components.postprocess.PostProcess import reconstruct

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model = AlprModel().to(device)
model.eval()

state_dict = torch.load("./model_1.pth", map_location=device)
model.load_state_dict(state_dict)

image = cv2.imread("./test_data/image.png")
resized = cv2.resize(image, (384, 384))
model_input = image.astype(np.float32)
model_input = cv2.resize(model_input, (384, 384))
model_input = model_input.transpose((2, 0, 1))
model_input = model_input / 255.0
model_input = model_input.reshape(1, 3, 384, 384)
model_input = torch.from_numpy(model_input).float().to(device)
prob, bbox = model(model_input)
predict_feature_map = torch.cat([prob, bbox], dim=1)
print(f"==>> prob.shape: {prob.shape}")
print(f"==>> affine.shape: {bbox.shape}")

reconstruct_result = reconstruct(image, resized, predict_feature_map, 0.5)
print(f"==>> reconstruct_result: {reconstruct_result}")