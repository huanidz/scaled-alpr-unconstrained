import numpy as np
import torch
# Literally adopt from original repo

def reconstruct(original_image, resized_image, predict_feature_map, output_shape, detection_threshold):
    
    net_stride 	= 2**4
    side 		= ((208. + 40.)/2.)/net_stride # 7.75

    predict_feature_map = predict_feature_map.squeeze()

    Probs = predict_feature_map[0, ...]
    Affines = predict_feature_map[2:, ...]
    
    xx, yy = np.where(Probs > detection_threshold)
    
    
 
 

