import cv2
import numpy as np
import os

def horizontal_flip(p, img, boxe):
    if np.random.random() > p:
        boxe[]
        img = cv2.flip(img, 1)
    return img

def vertical_flip(p, img, boxe):
    if np.random.random() > p:
        img = cv2.flip(img, 0)
    return img

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

if __name__ == '__main__':

    img_path = './images'
    augmentation_path = './check_augmentation'

    img_list = [f'{img_path}/{img}' for img in os.listdir(img_path)]

    if not os.path.exists(augmentation_path):
        os.mkdir(augmentation_path)

    for i, img in enumerate(img_list[:100]):

        img = cv2.imread(img)

        img = horizontal_flip(0.5, img)
        img = vertical_flip(0.5, img)

        cv2.imwrite(f'{augmentation_path}/{i}.jpeg', img)
