import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset
from utils.helper_func import IOU_centre_and_dims

# Augmentation
import random
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
import albumentations as alb

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
        
        self.counter = 0
        
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
        
        # Find center point of a contour which has points of the polygon
        contour = polygon.coords
        
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        Center_Of_Image = (W / 2, H / 2)
        offset_to_shift_X = int(Center_Of_Image[0] - cx)
        offset_to_shift_Y = int(Center_Of_Image[1] - cy)     
        
        translation_matrix = np.float32([[1, 0, offset_to_shift_X], [0, 1, offset_to_shift_Y]])
        image = cv2.warpAffine(image, translation_matrix, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        p1 = (p1[0] + offset_to_shift_X, p1[1] + offset_to_shift_Y)
        p2 = (p2[0] + offset_to_shift_X, p2[1] + offset_to_shift_Y)
        p3 = (p3[0] + offset_to_shift_X, p3[1] + offset_to_shift_Y)
        p4 = (p4[0] + offset_to_shift_X, p4[1] + offset_to_shift_Y)
        
        polygon = Polygon([p1, p2, p3, p4])
        polygons = PolygonsOnImage([polygon], shape=image.shape)
        
        # cv2.imwrite(f"/home/huan/prjdir/scaled-alpr-unconstrained/visual/warp.jpg", image)
        
        
        # TODO: Add transform / augmentation code here
        augmenter = iaa.Sequential([
            iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)}),
            iaa.Rotate((-45, 45)),  # Random rotation
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)}),  # Random shift
            iaa.Fliplr(0.5),  # Random horizontal flip
            iaa.GaussianBlur(sigma=(0, 1.5)),  # Random gaussian blur
            iaa.PerspectiveTransform(scale=(0.05, 0.1)),
            iaa.ChannelShuffle(0.2),  # Random channel shuffle
        ], random_order=True)
        
        cropper = iaa.CropToFixedSize(height=self.input_size, width=self.input_size, position="center")
        
        image = cropper.augment_image(image)
        resized_polys = cropper.augment_polygons(polygons)
        # cv2.imwrite(f"/home/huan/prjdir/scaled-alpr-unconstrained/visual/cropped_image_{self.counter}.jpg", image)
        
        augmenter = augmenter.to_deterministic()
        augmented_image = augmenter.augment_image(image)
        # cv2.imwrite(f"/home/huan/prjdir/scaled-alpr-unconstrained/visual/augmented_image_{self.counter}.jpg", augmented_image)
        
        # draw_img = augmented_image.copy()
        
        augmented_poly = augmenter.augment_polygons(resized_polys)[0].coords
        # int_aug_poly = np.int32(augmented_poly)
        
        # cv2.line(draw_img, tuple(int_aug_poly[0]), tuple(int_aug_poly[1]), (0, 0, 255), 2)
        # cv2.line(draw_img, tuple(int_aug_poly[1]), tuple(int_aug_poly[2]), (0, 0, 255), 2)
        # cv2.line(draw_img, tuple(int_aug_poly[2]), tuple(int_aug_poly[3]), (0, 0, 255), 2)
        # cv2.line(draw_img, tuple(int_aug_poly[3]), tuple(int_aug_poly[0]), (0, 0, 255), 2)
        
        # cv2.imwrite(f"/home/huan/prjdir/scaled-alpr-unconstrained/visual/augmented_poly_{self.counter}.jpg", draw_img)
        self.counter += 1
        
        
        bounding_rect = cv2.boundingRect(augmented_poly.astype(np.int32))
        
        output_feature_map = self._to_output_feature_map(bounding_rect, augmented_poly / self.input_size)
        output_feature_map = torch.from_numpy(output_feature_map).float()
                
        image = torch.from_numpy(augmented_image).permute(2, 0, 1).float().div(255.0)
        
        return image, output_feature_map
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        # image = cv2.resize(image, (self.input_size, self.input_size))
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
        num_ones = np.count_nonzero(Y == 1.0)
        # print(f"==>> num_ones: {num_ones}")
        return Y