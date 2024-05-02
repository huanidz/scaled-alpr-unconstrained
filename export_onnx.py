import torch
import torch.onnx

import torch
import torch.nn as nn
from components.model.AlprModel import AlprModel

# Path to your PyTorch checkpoint
PATH_TO_CHECKPOINT = "./checkpoints/small_best.pth"

model = AlprModel(scale="small")

# Load your PyTorch model
checkpoints = torch.load(PATH_TO_CHECKPOINT)
model.load_state_dict(checkpoints['model_state_dict'])

print("loaded model state dict")


# Set the model to evaluation mode
model.eval()

# Define a custom PyTorch module to fuse the output heads
class FusedAlprModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output1, output2 = self.model(x)
        fused_output = torch.cat((output1, output2), dim=1)  # Concatenate along channel dimension
        return fused_output

# Set input dimensions
input_shape = (1, 3, 384, 384)  # (batch_size, channels, height, width)

fused_model = FusedAlprModel(model)

# Set the input sample
dummy_input = torch.randn(input_shape)

# Export the model to ONNX format
print("Converting to onnx...")
output_path = "model.onnx"
torch.onnx.export(
    fused_model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

print(f"Model exported to {output_path}")