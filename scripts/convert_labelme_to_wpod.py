import os
import json
import argparse
import shutil
import numpy as np

def normalize_point(point, img_width, img_height):
    x, y = point
    return x / img_width, y / img_height

def order_points_clockwise(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)

    def angle(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])
    
    pts = sorted(pts, key=angle)
    pts = np.array(pts)

    # Ensure first point is top-left (min sum of x+y)
    min_index = np.argmin(pts.sum(axis=1))
    pts = np.roll(pts, -min_index, axis=0)

    return pts.tolist()

def convert_annotation(json_path, dst_folder):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]
    img_name = data["imagePath"]
    shapes = data["shapes"]

    annotations = []

    for shape in shapes:
        if shape["shape_type"] != "polygon":
            continue

        points = shape["points"]
        norm_points = [normalize_point(p, img_width, img_height) for p in points]
        ordered_points = order_points_clockwise(norm_points)

        xs = [p[0] for p in ordered_points]
        ys = [p[1] for p in ordered_points]

        result = xs + ys
        formatted = ", ".join(f"{val:.6f}" for val in result)
        annotations.append(formatted)

    # Write output .txt
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    txt_output_path = os.path.join(dst_folder, base_name + ".txt")
    with open(txt_output_path, "w") as f:
        for line in annotations:
            f.write(line + "\n")

    # Copy image
    src_img_path = os.path.join(os.path.dirname(json_path), img_name)
    dst_img_path = os.path.join(dst_folder, img_name)
    if os.path.exists(src_img_path):
        shutil.copy2(src_img_path, dst_img_path)

def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to custom format.")
    parser.add_argument("--src", required=True, help="Source folder with LabelMe JSON files and images.")
    parser.add_argument("--dst", required=True, help="Destination folder for images and converted .txt annotations.")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)

    for filename in os.listdir(args.src):
        if filename.endswith(".json"):
            json_path = os.path.join(args.src, filename)
            convert_annotation(json_path, args.dst)

if __name__ == "__main__":
    main()
