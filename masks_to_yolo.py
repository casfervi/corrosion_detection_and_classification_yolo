# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:01:03 2026

@author: vinicius.ferreira
"""
import cv2
import numpy as np
from pathlib import Path

for split in ["train", "test"]:

    mask_dir = Path(f"dataset/{split}/masks")
    label_dir = Path(f"dataset/{split}/labels")

    label_dir.mkdir(parents=True, exist_ok=True)
    
    for mask_file in mask_dir.glob("*.png"):
    
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    
        if mask is None:
            continue
    
        h, w = mask.shape
    
        txt_lines = []
    
        class_map = {
            38: 0,   # Fair
            75: 1,   # Poor
            113: 2   # Severe
        }
        
        for pixel_value, yolo_class in class_map.items():
            binary = (mask == pixel_value).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
    
            for cnt in contours:
    
                # Remove ruído muito pequeno
                area = cv2.contourArea(cnt)
    
                if area < 20:
                    continue
    
                # Simplifica o contorno
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
    
                if len(cnt) < 3:
                    continue
    
                points = []
    
                for point in cnt:
    
                    x = point[0][0] / w
                    y = point[0][1] / h
    
                    points.extend([
                        f"{x:.6f}",
                        f"{y:.6f}"
                    ])
    
                txt_lines.append(
                    f"{yolo_class} " +
                    " ".join(points)
                )
    
        output_file = label_dir / f"{mask_file.stem}.txt"
    
        with open(output_file, "w") as f:
            f.write("\n".join(txt_lines))