import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset
from utils.helper_func import IOU_centre_and_dims

# Augmentation
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

class AlprDataset(Dataset):
    
    def __init__(self, images_folder, labels_folder, input_size=384) -> None:
        super(AlprDataset, self).__init__()
        
        if images_folder[-1] == "/":
            images_folder = images_folder[:-1]
            
        if labels_folder[-1] == "/":
            labels_folder = labels_folder[:-1]
        
        self.images = natsorted(glob(f"{images_folder}/*"))
        self.labels = natsorted(glob(f"{labels_folder}/*"))
        self.input_size = input_size
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.load_image(self.images[index])
        label = self.load_label(self.labels[index])
        
        H, W, C = image.shape
        
        label[:4] *= W
        label[4:] *= H
        label = label.astype(np.int32)
        
        p1 = (label[0], label[4])
        p2 = (label[1], label[5])
        p3 = (label[2], label[6])
        p4 = (label[3], label[7])

        polygon = Polygon([p1, p2, p3, p4])
        polygons = PolygonsOnImage([polygon], shape=image.shape)
        
        # TODO: Add transform / augmentation code here
        augmenter = iaa.Sequential([
            iaa.Affine(scale={"x": (0.65, 1.75), "y": (0.65, 1.75)}),
            iaa.Rotate((-60, 60)),  # Random rotation
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)}),  # Random shift
            iaa.Fliplr(0.5),  # Random horizontal flip
            iaa.GaussianBlur(sigma=(0, 1.5)),  # Random gaussian blur
            iaa.AddToHueAndSaturation((-10, 10)),  # Random color change
            iaa.AdditiveGaussianNoise(scale=0.01*255),  # Random noise
            iaa.ChannelShuffle(0.1),  # Random channel shuffle
        ], random_order=True)
        
        augmenter = augmenter.to_deterministic()
        augmented_image = augmenter.augment_image(image)
        augmented_poly = augmenter.augment_polygons(polygons)[0].coords
        bounding_rect = cv2.boundingRect(augmented_poly.astype(np.int32))
        
        output_feature_map = self._to_output_feature_map(bounding_rect, augmented_poly / self.input_size)
                
        image = torch.from_numpy(augmented_image).permute(2, 0, 1).float().div(255.0)
        
        return image, output_feature_map
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.input_size, self.input_size))
        return image
    
    def load_label(self, label_path):
        label = np.genfromtxt(label_path, delimiter=",") # Label has format of: x1, x2, y1, y2, x3, y3, x4, y4 (x1, y1 is top-left, and the order is clockwise)
        label = np.array(label, dtype=np.float32)   
        
        return label
    
    def _to_output_feature_map(self, gt_plate_bounding_rect, gt_plate_corner_pts):
        print(f"==>> gt_plate_corner_pts: {gt_plate_corner_pts}")
        gt_plate_bounding_rect = np.array(gt_plate_bounding_rect) / self.input_size
        output_scale = 16 # Stride = 16
        
        side = ((float(self.input_size) + 40.0) / 2.0)/ output_scale 
        outsize = self.input_size // output_scale
        
        Y  = np.zeros((2*4+1, outsize, outsize), dtype=np.float32)
        MN = np.array([outsize, outsize], dtype=np.float32)
        WH = np.array([self.input_size, self.input_size], dtype=np.float32)

        gt_plate_brect_topleft = gt_plate_bounding_rect[0], gt_plate_bounding_rect[1] # x, y of ground truth plate's boudning rect, in 0-->1 range respect to image
        gt_plate_brect_bottomright = gt_plate_bounding_rect[0] + gt_plate_bounding_rect[2], gt_plate_bounding_rect[1] + gt_plate_bounding_rect[3] # x, y of ground truth plate's boudning rect, in 0-->1 range respect to image

        gt_plate_brect_wh = (gt_plate_bounding_rect[2] - gt_plate_bounding_rect[3])
        print(f"==>> gt_plate_brect_wh: {gt_plate_brect_wh}")
        gt_plate_brect_center = gt_plate_bounding_rect[0] + gt_plate_bounding_rect[2]/2, gt_plate_bounding_rect[1] + gt_plate_bounding_rect[3]/2

        tlx, tly = np.floor(np.maximum(gt_plate_brect_topleft, 0.0) * MN).astype(int).tolist()
        brx, bry = np.ceil(np.minimum(gt_plate_brect_bottomright, 1.0) * MN).astype(int).tolist()

        for x in range(tlx, brx):
            
            for y in range(tly, bry):

                mn = np.array([float(x) + .5, float(y) + .5])
                print(f"==>> mn: {mn}")
                iou = IOU_centre_and_dims(mn/MN, gt_plate_brect_wh, gt_plate_brect_center, gt_plate_brect_wh)
                print(f"==>> iou: {iou}")

                if iou > 0.5:
                    p_WH = gt_plate_corner_pts * WH.reshape((2, 1))
                    print(f"==>> p_WH: {p_WH}")
                    p_MN = p_WH / output_scale

                    p_MN_center_mn = p_MN - mn.reshape((2,1))

                    p_side = p_MN_center_mn / side

                    Y[0, y, x] = 1.
                    Y[1:, y, x] = p_side.T.flatten()

        return Y