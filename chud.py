import torch
import numpy as np

a = np.array([[1,2,3],[4,5,6]])

b = np.where(a > 2)
# print(f"==>> b: {b}")

# Make a dummy 4x4x3 tensor with integer values between 0 and 100
c = np.random.randint(0, 101, size=(4, 4, 3))
# c = np.random.rand(4, 4, 3)
d = c[:, 2, 2]

# Random tensor with value range from -1 to 1
d = np.random.randn(2, 4, 4)
print(f"==>> d: {d}")

e = torch.softmax(torch.from_numpy(d), dim=0)
print(f"==>> e: {e}")

