import os
import cv2
import torch
import numpy as np
from components.model.AlprModel import AlprModel
from components.processes.InferenceProcess import preprocess, reconstruct

import argparse
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="File path to image", required=True)
parser.add_argument("-s", "--size", help="Size of input image", default=384, type=int)
parser.add_argument("-t", "--threshold", help="Detection threshold", default=0.5, type=float)
args = parser.parse_args()

# Default to CPU since it's fast enough to run a demo!
device = torch.device("cpu")

model = AlprModel().to(device)
model.eval()

state_dict = torch.load("./checkpoints/model_69.pth", map_location=device)
model.load_state_dict(state_dict)

image = cv2.imread(args.image)
H, W, C = image.shape

print("Inference single image...")
start_infer = perf_counter()

resized, model_input = preprocess(image, args.size)
with torch.no_grad():
    prob, bbox = model(model_input)
    
predict_feature_map = torch.cat([prob, bbox], dim=1).detach().cpu()

results = reconstruct(resized, predict_feature_map, args.threshold)

if len(results) == 0:
    print("No plates found!")
    exit()
    
end_infer = perf_counter()
print("Time taken: ", end_infer - start_infer, "ms")
print("Total plates found: ", len(results))

for plate in results:
    coordinates = plate[0].numpy().transpose((1, 0))
    probs = plate[1].numpy()

    coordinates *= np.array([W, H])
    coordinates = coordinates.astype(np.int32)

    cv2.polylines(image, [coordinates], True, (0, 255, 0), 2)

# Save result
if not os.path.exists("./inference"):
    os.mkdir("./inference")
    
cv2.imwrite("./inference/inference.jpg", image)
print("Saving result image to inference.jpg")