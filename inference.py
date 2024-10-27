import os
import cv2
import torch
import numpy as np
from components.model.AlprModel import AlprModel
from components.processes.InferenceProcess import preprocess, reconstruct

import torch.cuda.amp as amp
from torch.cuda.amp import autocast

import argparse
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="Path to model checkpoint (pth)", default=None, type=str)
parser.add_argument("--size", help="Size of input image", default=384, type=int)
parser.add_argument("--threshold", help="Detection threshold", default=0.7, type=float)
parser.add_argument("--scale", help="Scale of the model (tiny, small, base, large)", default="base", type=str)
args = parser.parse_args()

# Default to CPU since it's fast enough to run a demo!
device = torch.device("cuda")

model = AlprModel(scale=args.scale).to(device)
# model = torch.compile(model)
model.eval()

# try:
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# except:
#     print("No model checkpoint found. please enter path to valid checkpoint...")
#     exit()

# image = cv2.imread(args.image)
image = cv2.imread(input("Enter image path: "))
H, W, C = image.shape

print("Inference single image...")
start_infer = perf_counter()

resized, model_input = preprocess(image, args.size)
model_input = model_input.to(device=device)
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

for i, plate in enumerate(results):
    print(f"  Plate: {i + 1}")
    print(f"    Score: {plate[1].numpy()}")
    print(f"    Coordinates: {plate[0].numpy().transpose((1, 0))}")
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

# Display
display_resized = cv2.resize(image, (800, 600))
cv2.imshow("Result", display_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

