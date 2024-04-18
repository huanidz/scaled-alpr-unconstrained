import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from components.model.AlprModel import AlprModel
from components.losses.loss import AlprLoss
from components.data.AlprData import AlprDataset 
from time import perf_counter
import numpy as np
import albumentations as A
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import cv2

loader = AlprDataset(images_folder="./train_data/images", labels_folder="./train_data/labels", input_size=384)
image, label = loader[0]


# # Load an image and its polygon labels
# image = cv2.imread('./train_data/images/00011.jpg')
# image = cv2.resize(image, (384, 384))
# H, W, C = image.shape

# p1 = (label[0], label[4])
# p2 = (label[1], label[5])
# p3 = (label[2], label[6])
# p4 = (label[3], label[7])

# polygon = Polygon([p1, p2, p3, p4])

# # Define the transformations
# augmenter = iaa.Sequential([
#     iaa.Affine(scale={"x": (0.65, 1.75), "y": (0.65, 1.75)}),
#     iaa.Rotate((-60, 60)),  # Random rotation up to 30 degrees
#     iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)}),  # Random shift
#     iaa.Fliplr(0.5),  # Random horizontal flip
#     iaa.GaussianBlur(sigma=(0, 1.5)),  # Random gaussian blur
#     iaa.AddToHueAndSaturation((-10, 10)),  # Random color change
#     iaa.AdditiveGaussianNoise(scale=0.01*255),  # Random noise
#     iaa.ChannelShuffle(0.1),  # Random channel shuffle
# ], random_order=True)

# # Apply the transformations
# # augmented_image, augmented_polygons = augmenter(image=image, polygons=reshaped_polygons)
# polygons = PolygonsOnImage([polygon], shape=image.shape)
# augmenter = augmenter.to_deterministic()
# augmented_image = augmenter.augment_image(image)
# augmented_poly = augmenter.augment_polygons(polygons)[0].coords

# bounding_rect = cv2.boundingRect(augmented_poly.astype(np.int32))


# # Display the augmented image and polygons
# augmented_image = np.ascontiguousarray(augmented_image).astype(np.uint8)
# cv2.polylines(augmented_image, [augmented_poly.astype(np.int32)], True, (0, 255, 0), 2)
# cv2.rectangle(augmented_image, (bounding_rect[0], bounding_rect[1]), (bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3]), (0, 0, 255), 2)
# cv2.imshow('Augmented Image', augmented_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
