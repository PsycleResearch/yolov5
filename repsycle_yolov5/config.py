import torch

image_size = 640
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image size / stride = nb_pixel per cell for three yolo scales
scales = [image_size // 32, image_size // 16, image_size // 8]

nb_classes = 1

anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
# class names
class_names = ['copper']
