import cv2
import numpy as np

# Load the image
img = cv2.imread('./train_data/images/00011.jpg')

# Define the shift offsets (x, y)
x_offset = 50
y_offset = -30

# Get the image dimensions
height, width = img.shape[:2]

# Create the translation matrix
translation_matrix = np.float32([[1, 0, x_offset], [0, 1, y_offset]])

# Warp the image using the translation matrix
shifted_img = cv2.warpAffine(img, translation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Display the shifted image
cv2.imshow('Shifted Image', shifted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()