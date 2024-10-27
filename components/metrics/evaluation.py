from shapely.geometry import Polygon
from shapely.validation import make_valid

def polygon_area(poly):
    """
    Calculate the area of a polygon given its vertices.
    """
    return Polygon(poly).area

def intersection_area(poly1, poly2):
    """
    Calculate the intersection area between two polygons.
    """
    poly1 = make_valid(Polygon(poly1))
    poly2 = make_valid(Polygon(poly2))
    
    return poly1.intersection(poly2).area

def calculate_metrics(pred_poly, gt_poly):
    """
    Calculate IoU, Dice coefficient, and F1-score for polygon evaluation.
    
    Args:
        pred_poly (list): List of vertices representing the predicted polygon.
        gt_poly (list): List of vertices representing the ground truth polygon.
        
    Returns:
        tuple: IoU, and F1-score.
    """
    pred_area = polygon_area(pred_poly)
    gt_area = polygon_area(gt_poly)
    intersection = intersection_area(pred_poly, gt_poly)
    
    # Calculate IoU
    union = pred_area + gt_area - intersection
    iou = intersection / union
    
    # Calculate F1-score
    precision = intersection / pred_area if pred_area > 0 else 0
    recall = intersection / gt_area if gt_area > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return iou, f1

# Example usage
# pred_poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
# gt_poly = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]

# iou, dice, f1 = calculate_metrics(pred_poly, gt_poly)
# print(f"IoU: {iou:.4f}")
# print(f"F1-score: {f1:.4f}")