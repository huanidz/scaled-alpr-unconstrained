import numpy as np

def IoU(tl_rect_A, br_rect_A, tl_rect_B, br_rect_B):

        WH_Rect_A = br_rect_A - tl_rect_A
        WH_Rect_B = br_rect_B - tl_rect_B
	
        intersection_wh = np.maximum(np.minimum(br_rect_A, br_rect_B) - np.maximum(tl_rect_A, tl_rect_B), 0.)
        intersection_area = np.prod(intersection_wh)
    
        area1, area2 = (np.prod(WH_Rect_A), np.prod(WH_Rect_B))
        union_area = area1 + area2 - intersection_area
    
        return intersection_area / union_area

def IOU_centre_and_dims(mn_center, mn_rect_wh, bounding_rect_center, bounding_rect_wh):
        print(f"==>> mn_center: {mn_center}")
        print(f"==>> mn_rect_wh: {mn_rect_wh}")
        print(f"==>> bounding_rect_center: {bounding_rect_center}")
        print(f"==>> bounding_rect_wh: {bounding_rect_wh}")
        print()
        return IoU(mn_center - mn_rect_wh / 2, \
                        mn_center + mn_rect_wh / 2, \
                        bounding_rect_center - bounding_rect_wh / 2, \
                        bounding_rect_center + bounding_rect_wh / 2)
