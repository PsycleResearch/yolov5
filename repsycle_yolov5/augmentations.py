import cv2
import numpy as np
import os
from utils import plot_images
import dataset

def horizontal_flip(p, img, boxes):

    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)

    if np.random.random() > p:

        boxes[:,0] = 1 - boxes[:,0]
        img = cv2.flip(img, 1)

    return img, boxes

def vertical_flip(p, img, boxes):

    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes)

    if np.random.random() > p:
        boxes[:, 0] = 1 - boxes[:, 0]
        img = cv2.flip(img, 1)

    return img, boxes

def gaussian_noise(img, p, mean, sigma):
    if np.random.random() > p:
            gaussian = np.random.normal(mean, sigma, img.shape).astype('uint8')
            img = cv2.add(img, gaussian)
    return img

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):

    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    return img_hsv

if __name__ == '__main__':

    img_path = './images'
    augmentation_path = './check_augmentation'

    import json
    with open('datas/training_set.json') as f:
        annotations = json.load(f)

    if not os.path.exists(augmentation_path):
        os.mkdir(augmentation_path)

    for i, (img_id, boxes) in enumerate(annotations.items()):

        img = cv2.imread(f'{img_path}/{img_id}.jpeg')

        img, boxes = horizontal_flip(0.5, img, boxes)
        img, boxes = vertical_flip(0.5, img, boxes)
        img = gaussian_noise(img, 0.5, 0, 0.5)
        #img = augment_hsv(img, hgain=0.01, sgain=0.01, vgain=0.01)

        plot_images(img, boxes)

        # cv2.imwrite(f'{augmentation_path}/{i}.jpeg', img)
