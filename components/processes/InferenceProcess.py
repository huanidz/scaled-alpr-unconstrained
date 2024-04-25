import cv2
import torch
import numpy as np
from utils.helper_func import nms

def preprocess(image, target_size):
    image_resized = cv2.resize(image, (target_size, target_size))
    model_input = image.astype(np.float32)
    model_input = cv2.resize(model_input, (target_size, target_size))
    model_input = model_input.transpose((2, 0, 1))
    model_input = model_input / 255.0
    model_input = model_input.reshape(1, 3, target_size, target_size)
    model_input = torch.from_numpy(model_input).float()
    return image_resized, model_input

def reconstruct(resized_image, predict_feature_map, detection_threshold):
    """
    Adopt from original repo and simplified!
    """
    resized_H, resized_W, _ = resized_image.shape
    
    net_stride 	= 16 # the downsample scale of the model
    balance_side = 45.0 # 45 is my choose for 384x384 input size, 40 is for 208 from the original, adjust it acordingly to your input size
    side 		= ((resized_W + balance_side) / 2.0)/net_stride
    
    predict_feature_map = predict_feature_map.squeeze() # 9x24x24

    Probs = predict_feature_map[0, ...] # 24x24 (first dim), The predict_feature_map[1, ...] is non-obj prob which is not used in post-process
    Affines = predict_feature_map[2:, ...] # 6x24x24 (last dims from the second dim)
    
    yy_valid, xx_valid = np.where(Probs > detection_threshold)
    
    H_resized, W_resized, _ = resized_image.shape
    MN = torch.Tensor([W_resized / net_stride, H_resized / net_stride])
    
    vxx = 0.47 # alpha
    vyy = 0.495 # alpha
    
    base = torch.Tensor([[-vxx, -vyy, 1.],
                      [ vxx, -vyy, 1.],
                      [ vxx,  vyy, 1.],
                      [-vxx,  vyy, 1.]]).T
    
    labels = [] # labels here mean raw plates that need to put into nms
    for i in range(len(yy_valid)):
        y, x = yy_valid[i], xx_valid[i] # Potential bug
        affine = Affines[:, y, x] # (6, )
        prob = Probs[y, x] # (1, )
 
        mn = torch.Tensor([float(x) + 0.5, float(y) + 0.5])

        A = torch.reshape(affine,(2,3))
        A[0,0] = max(A[0,0], 0.)
        A[1,1] = max(A[1,1], 0.)

        pts = torch.Tensor(A @ base)
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2,1))

        pts_prop = pts_MN / MN.reshape((2,1))

        if prob > 0 and prob < 1:
            labels.append((pts_prop, prob))

    final_labels = nms(Labels=labels, iou_threshold=0.1)
    TLps = []
    
    if len(final_labels) > 0:
        final_labels.sort(key=lambda l: l[1], reverse=True) # Sort by prob
        
        for i, label in enumerate(final_labels):
            TLps.append((label))
            
    return TLps
    
    
    
 
 

