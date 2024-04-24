import cv2
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
        return IoU(mn_center - mn_rect_wh / 2, \
                        mn_center + mn_rect_wh / 2, \
                        bounding_rect_center - bounding_rect_wh / 2, \
                        bounding_rect_center + bounding_rect_wh / 2)

def IOU_labels(l1, l2):
    # L: [[x1, x2, x3, x4], [y1, y2, y3, y4]]
    
    l1_pts = l1[0].numpy().transpose(1, 0)
    l2_pts = l2[0].numpy().transpose(1, 0)
    
    # Finding bounding rect
    x_coords_1 = [point[0] for point in l1_pts]
    y_coords_1 = [point[1] for point in l1_pts]
    
    x_min_1, x_max_1 = min(x_coords_1), max(x_coords_1)
    y_min_1, y_max_1 = min(y_coords_1), max(y_coords_1)
    
    top_left_1 = np.array([x_min_1, y_min_1], np.float32)
    bottom_right_1 = np.array([x_max_1, y_max_1], np.float32)
    
    x_coords_2 = [point[0] for point in l2_pts]
    y_coords_2 = [point[1] for point in l2_pts]
    
    x_min_2, x_max_2 = min(x_coords_2), max(x_coords_2)
    y_min_2, y_max_2 = min(y_coords_2), max(y_coords_2)
    
    top_left_2 = np.array([x_min_2, y_min_2], np.float32)
    bottom_right_2 = np.array([x_max_2, y_max_2], np.float32)
    
    return IoU(top_left_1, bottom_right_1, top_left_2, bottom_right_2)
     
def nms(Labels, iou_threshold = 0.5):
    SelectedLabels = []
    # each item in Labels: (pts, prob)
    Labels.sort(key=lambda l: l[1],reverse=True)

    for label in Labels:

        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)

    return SelectedLabels
   
def crop_and_pad(image, x, y, w_range, h_range, polygon_coords=None):
        # Get the dimensions of the input image
        height, width = image.shape[:2]

        w = int(width * np.random.uniform(w_range[0], w_range[1]))
        h = int(height * np.random.uniform(h_range[0], h_range[1]))

        # Calculate the coordinates of the crop box
        x1 = max(x - w // 2, 0)
        y1 = max(y - h // 2, 0)
        x2 = min(x + w // 2, width)
        y2 = min(y + h // 2, height)

        # Crop the image
        cropped = image[y1:y2, x1:x2]

        # Calculate the padding required
        pad_left = abs(min(0, x - w // 2))
        pad_right = abs(max(0, x + w // 2 - width))
        pad_top = abs(min(0, y - h // 2))
        pad_bottom = abs(max(0, y + h // 2 - height))

        # Pad the cropped image with black pixels
        padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Adjust the polygon coordinates if provided
        if polygon_coords is not None:
            adjusted_coords = []
            for px, py in polygon_coords:
                # Adjust the x-coordinate
                if px < x1:
                    adj_x = pad_left
                elif px > x2:
                    adj_x = w - pad_right
                else:
                    adj_x = px - x1 + pad_left

                # Adjust the y-coordinate
                if py < y1:
                    adj_y = pad_top
                elif py > y2:
                    adj_y = h - pad_bottom
                else:
                    adj_y = py - y1 + pad_top

                adjusted_coords.append((adj_x, adj_y))
        else:
            adjusted_coords = None

        return padded, adjusted_coords
