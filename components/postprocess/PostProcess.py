import numpy as np
import torch
from utils.helper_func import nms
# Literally adopt from original repo


def get_pts_homogeneous(tlx, tly, brx, bry):
    """
        Alias for getRectPts in original repo
    """    
    return np.array([
        [tlx, brx, brx, tlx],
        [tly, tly, bry, bry],
        [1.0, 1.0, 1.0, 1.0]
    ], np.float32)

def reconstruct(original_image, resized_image, predict_feature_map, detection_threshold):
    
    resized_H, resized_W, _ = resized_image.shape
    
    net_stride 	= 16 # 2**4 
    side 		= ((resized_W + 40.)/2.)/net_stride # 7.75
    
    predict_feature_map = predict_feature_map.squeeze() # 9x24x24

    Probs = predict_feature_map[0, ...] # 24x24 (first dim), The predict_feature_map[1, ...] is non-obj prob which is not used in post-process
    Affines = predict_feature_map[2:, ...] # 6x24x24 (last dims from the second dim)
    
    yy_valid, xx_valid = np.where(Probs > detection_threshold)
    
    H_resized, W_resized, _ = resized_image.shape
    MN = torch.Tensor([W_resized / net_stride, H_resized / net_stride])
    
    vxx = vyy = 0.5 # alpha
    
    base = torch.Tensor([[-vxx, -vyy, 1.],
                      [ vxx, -vyy, 1.],
                      [ vxx,  vyy, 1.],
                      [-vxx,  vyy, 1.]]).T
    labels = []

    for i in range(len(yy_valid)):
        y, x = yy_valid[i], xx_valid[i] # Potential bug
        affine = Affines[:, y, x] # (6, )
        prob = Probs[y, x] # (1, )
 
        mn = torch.Tensor([float(x) + .5, float(y) + .5])

        A = torch.reshape(affine,(2,3))
        A[0,0] = max(A[0,0], 0.)
        A[1,1] = max(A[1,1], 0.)

        pts = torch.Tensor(A @ base)
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2,1))

        pts_prop = pts_MN / MN.reshape((2,1))

        # print(f"==>> pts_prop: {pts_prop}")
        # print(f"==>> prob: {prob}")
        # raise Exception("haha")
        if prob > 0 and prob < 1:
            labels.append((pts_prop, prob))

    print(f"==>> labels: {len(labels)}")
    final_labels = nms(Labels=labels, iou_threshold=0.1)
    TLps = []
    
    if len(final_labels) > 0:
        final_labels.sort(key=lambda l: l[1], reverse=True) # Sort by prob
        
        for i, label in enumerate(final_labels):
            TLps.append((label))
            
    return TLps
    
    
    
 
 

