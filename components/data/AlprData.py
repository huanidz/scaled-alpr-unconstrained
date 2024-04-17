import cv2
import torch
import numpy as np
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset
from utils.helper_func import IOU_centre_and_dims

class AlprDataset(Dataset):
    
    def __init__(self, images_folder, labels_folder, input_size=384) -> None:
        super(AlprDataset, self).__init__()
        
        self.images = natsorted(glob(f"{images_folder}/*"))
        self.labels = natsorted(glob(f"{labels_folder}/*"))
        self.input_size = input_size
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.load_image(self.images[index])
        label = self.load_label(self.labels[index])
        target_feature_map = self._to_output_feature_map(label[0], label[1:])
        # TODO: Add transform / augmentation code here
        
        
        
        return image, label
    
    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        return image
    
    def load_label(self, label_path):
        label = np.genfromtxt(label_path, delimiter=",") # Label has format of: x1, x2, y1, y2, x3, y3, x4, y4 (x1, y1 is top-left, and the order is clockwise)
        label = np.array([label * self.input_size], dtype=np.uint8)   
        print(f"==>> label: {label}")
        # label = torch.from_numpy(label).float()
        
        # Get the bounding rect from polygon made of these label points
        x, y, w, h = cv2.boundingRect(label)
        print(f"==>> x: {x}")
        print(f"==>> y: {y}")
        print(f"==>> w: {w}")
        print(f"==>> h: {h}")
        # gt_plate_bounding_rect = np.array([rect[0]/self.input_size, rect[1]/self.input_size, (rect[0]+rect[2])/self.input_size, (rect[1]+rect[3])/self.input_size])
        
        return label
    
    def _to_output_feature_map(self, gt_plate_bounding_rect, gt_plate_corner_pts):
        output_scale = 16 # Stride = 16
        
        side = ((float(self.input_size) + 40.)/2.)/ output_scale 
        outsize = self.input_size // output_scale
        
        Y  = np.zeros((2*4+1, outsize, outsize), dtype=np.float32)
        MN = np.array([outsize, outsize], dtype=np.float32)
        WH = np.array([self.input_size, self.input_size], dtype=np.float32)

        gt_plate_brect_topleft = gt_plate_bounding_rect[0] # x, y of ground truth plate's boudning rect, in 0-->1 range respect to image
        gt_plate_brect_bottomright = gt_plate_bounding_rect[1] # x, y of ground truth plate's boudning rect, in 0-->1 range respect to image

        gt_plate_brect_wh = gt_plate_brect_bottomright - gt_plate_brect_topleft
        gt_plate_brect_center = (gt_plate_brect_topleft + gt_plate_brect_bottomright) / 2

        tlx, tly = np.floor(np.maximum(gt_plate_brect_topleft, 0.0) * MN).astype(int).tolist()
        brx, bry = np.ceil(np.minimum(gt_plate_brect_bottomright, 1.0) * MN).astype(int).tolist()

        for x in range(tlx, brx):
            
            for y in range(tly, bry):

                mn = np.array([float(x) + .5, float(y) + .5])
                iou = IOU_centre_and_dims(mn/MN, gt_plate_brect_wh, gt_plate_brect_center, gt_plate_brect_wh)

                if iou > .5:

                    p_WH = gt_plate_corner_pts * WH.reshape((2, 1))
                    p_MN = p_WH / output_scale

                    p_MN_center_mn = p_MN - mn.reshape((2,1))

                    p_side = p_MN_center_mn / side

                    Y[0, y, x] = 1.
                    Y[1:, y, x] = p_side.T.flatten()

        return Y