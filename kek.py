import numpy as np
import torch

# a = torch.randn(8, 6, 32, 32)
# print(f"==>> a.shape: {a.shape}")
# b = a[:, 0]
# c = a[:,0,:,:] + 2
# if torch.allclose(b, c):
#     print("asd")
# print(f"==>> b.shape: {b.shape}")

v = 0.5
base = np.stack([[[[-v,-v,1., v,-v,1., v,v,1., -v,v,1.]]]])
print(f"==>> base: {base}")
print(f"==>> base.shape: {base.shape}")



