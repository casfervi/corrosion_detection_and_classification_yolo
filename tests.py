# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 14:10:06 2026

@author: vinicius.ferreira
"""

# import cv2
# import numpy as np

# mask = cv2.imread(
#     "C:/Users/vinicius.ferreira/Documents/Projects/corrosao_visualization/treino/dataset_vt/Train/masks/12.png",
#     cv2.IMREAD_GRAYSCALE
# )

# print(mask.shape)
# print(np.unique(mask))

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# mask = cv2.imread(
#     r"C:/Users/vinicius.ferreira/Documents/Projects/corrosao_visualization/treino/dataset_vt/Train/masks/12.png",
#     cv2.IMREAD_GRAYSCALE
# )

# for value in [38, 75, 113]:
#     plt.figure()
#     plt.imshow(mask == value, cmap="gray")
#     plt.title(f"Classe {value}")
#     plt.show()

# from pathlib import Path

# labels = list(Path("dataset/Train/labels").glob("*.txt"))

# print("labels:", len(labels))

# empty = sum(f.stat().st_size == 0 for f in labels)

# print("vazios:", empty)

# from ultralytics import YOLO

# model = YOLO("yolo11n-seg.pt")

# # model.train(
# #     data="corrosion.yaml",
# #     epochs=1,
# #     imgsz=512,
# #     batch=4
# # )

# model.train(
#     data="corrosion.yaml",
#     epochs=50,
#     imgsz=512,
#     batch=8
# )

# from pathlib import Path

# labels = list(Path("dataset/test/labels").glob("*.txt"))

# print("labels:", len(labels))

# empty = sum(f.stat().st_size == 0 for f in labels)

# print("vazios:", empty)


from ultralytics import YOLO

model = YOLO(
    r"C:\Users\vinicius.ferreira\Documents\Projects\corrosao_visualization\yolov11\Industrial-Corrosion-Detection-YOLOv11\runs\segment\train-4\weights\best.pt"
)

results = model.predict(
    source=r"C:/Users/vinicius.ferreira/Documents/Projects/corrosao_visualization/treino/dataset/train/images/12.jpeg",
    save=True,
    conf=0.25
)

results[0].show()