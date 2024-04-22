import albumentations as A
import cv2
import numpy as np

image = cv2.imread("./train_data/images/00011.jpg")
label = np.genfromtxt("./train_data/labels/00011.txt", delimiter=",") # Label has format of: x1, x2, y1, y2, x3, y3, x4, y4 (x1, y1 is top-left, and the order is clockwise)
label = np.array(label, dtype=np.float32)

H, W, C = image.shape
        
label[:4] *= W
label[4:] *= H
label = label.astype(np.int32)

p1 = [label[0], label[4]]
print(f"==>> p1: {p1}")

p2 = [label[1], label[5]]
print(f"==>> p2: {p2}")

p3 = [label[2], label[6]]
print(f"==>> p3: {p3}")

p4 = [label[3], label[7]] 
print(f"==>> p4: {p4}")

polygons = [
    np.array([p1, p2, p3, p4], dtype=np.int32)
]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5)
])

augmented = transform(image=image, keypoints=polygons)
augmented_image = augmented['image']
augmented_polygons = augmented['polygons']

for polygon in augmented_polygons:
    polygon = polygon.astype(np.int32)
    cv2.polylines(augmented_image, [polygon.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

cv2.imshow('Augmented Image', augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
