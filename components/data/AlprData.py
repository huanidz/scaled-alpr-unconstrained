import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset
from shapely.geometry import Polygon as ShapelyPolygon
from utils.helper_func import IOU_centre_and_dims

# Augmentation
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

class AlprDataset(Dataset):
    
    def __init__(self, images_folder, labels_folder, input_size=384, mode="train") -> None:
        super(AlprDataset, self).__init__()
        
        if images_folder[-1] == "/":
            images_folder = images_folder[:-1]
            
        if labels_folder[-1] == "/":
            labels_folder = labels_folder[:-1]
        
        self.images = natsorted(glob(f"{images_folder}/*"))
        self.labels = natsorted(glob(f"{labels_folder}/*"))
        self.input_size = input_size
        self.mode = mode
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.load_image(self.images[index])
        label = self.load_label(self.labels[index])
        
        H, W, C = image.shape
        
        if self.mode == "eval":

            p1 = (label[0], label[4])
            p2 = (label[1], label[5])
            p3 = (label[2], label[6])
            p4 = (label[3], label[7])

            resized_image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
            model_input = torch.from_numpy(resized_image).permute(2, 0, 1).float().div(255.0)
            gt_plate_coordinate = np.array([p1, p2, p3, p4], np.float32) # ground truth
            return resized_image, model_input, gt_plate_coordinate
        
        label[:4] *= W
        label[4:] *= H
        label = label.astype(np.int32)
        
        
        p1 = (label[0], label[4])
        p2 = (label[1], label[5])
        p3 = (label[2], label[6])
        p4 = (label[3], label[7])


        polygon = Polygon([p1, p2, p3, p4])
        polygons = PolygonsOnImage([polygon], shape=image.shape)
        
        # Augmenter for image (after cropping/non-cropping)
        augmenter = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.CropAndPad(px=(-150, 150))),
            iaa.Rotate((-30, 30)),  # Random rotation
            iaa.Fliplr(0.5),  # Random horizontal flip
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),  # Random gaussian blur
            iaa.ChannelShuffle(0.1),  # Random channel shuffle,
            iaa.Sometimes(0.3, iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255))),  # Additive laplace noise
            iaa.Sometimes(0.4, iaa.MultiplySaturation((0.5, 1.5))),  # Adjust saturation
            iaa.Sometimes(0.3, iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)),  # Adjust hue and saturation
            iaa.Sometimes(0.4, iaa.MultiplyBrightness((0.6, 1.4))),
            iaa.Sometimes(0.3, iaa.Sharpen(alpha=(0.0, 0.8), lightness=(0.75, 1.5))),
            iaa.Resize({"height": self.input_size, "width": self.input_size})
        ])
        
        augmenter = augmenter.to_deterministic()
        augmented_image = augmenter.augment_image(image)
        augmented_poly = augmenter.augment_polygons(polygons)
        
        # Ensure the polys is inside the image, otherwise it will cause some error later
        if augmented_poly[0].is_out_of_image(augmented_image.shape, partly=True):
            return self.__getitem__(index)
        
        
        augmented_poly = augmenter.augment_polygons(polygons)[0].coords
        bounding_rect = cv2.boundingRect(augmented_poly.astype(np.int32))
        
        output_feature_map = self._to_output_feature_map(bounding_rect, augmented_poly / self.input_size)
        output_feature_map = torch.from_numpy(output_feature_map).float()
                
        image = torch.from_numpy(augmented_image).permute(2, 0, 1).float().div(255.0)
        
        return image, output_feature_map

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        return image
    
    def load_label(self, label_path):
        label = np.genfromtxt(label_path, delimiter=",") # Label has format of: x1, x2, y1, y2, x3, y3, x4, y4 (x1, y1 is top-left, and the order is clockwise)
        label = np.array(label, dtype=np.float32)   
        
        return label
    
    def _to_output_feature_map(self, gt_plate_bounding_rect, gt_plate_corner_pts):
        gt_plate_bounding_rect = np.array(gt_plate_bounding_rect) / self.input_size
        output_scale = 16 # Stride = 16
        
        side = ((float(self.input_size) + 40.0) / 2.0)/ output_scale 
        outsize = self.input_size // output_scale
        
        Y  = np.zeros((2*4+1, outsize, outsize), dtype=np.float32)
        MN = np.array([outsize, outsize], dtype=np.float32)
        WH = np.array([self.input_size, self.input_size], dtype=np.float32)

        gt_plate_brect_topleft = np.array([gt_plate_bounding_rect[0], gt_plate_bounding_rect[1]], np.float32) # x, y of ground truth plate's boudning rect, in 0-->1 range respect to image
        gt_plate_brect_bottomright = np.array([gt_plate_bounding_rect[0] + gt_plate_bounding_rect[2], gt_plate_bounding_rect[1] + gt_plate_bounding_rect[3]], np.float32) # x, y of ground truth plate's boudning rect, in 0-->1 range respect to image

        gt_plate_brect_wh = np.array([gt_plate_bounding_rect[2], gt_plate_bounding_rect[3]], np.float32)
        gt_plate_brect_center = np.array([gt_plate_bounding_rect[0] + gt_plate_bounding_rect[2]/2, gt_plate_bounding_rect[1] + gt_plate_bounding_rect[3]/2], np.float32)

        tlx, tly = np.floor(np.maximum(gt_plate_brect_topleft, 0.0) * MN).astype(int).tolist()
        brx, bry = np.ceil(np.minimum(gt_plate_brect_bottomright, 1.0) * MN).astype(int).tolist()

        for x in range(tlx, brx):
            
            for y in range(tly, bry):

                mn = np.array([float(x) + .5, float(y) + .5])
                iou = IOU_centre_and_dims(mn/MN, gt_plate_brect_wh, gt_plate_brect_center, gt_plate_brect_wh)

                if iou > 0.5:
                    p_WH = gt_plate_corner_pts * WH
                    p_MN = p_WH / output_scale

                    p_MN_center_mn = p_MN - mn

                    p_side = p_MN_center_mn / side

                    Y[0, y, x] = 1.0
                    Y[1:, y, x] = p_side.flatten()
        return Y