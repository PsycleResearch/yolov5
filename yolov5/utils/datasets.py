import math
import os
import random
from pathlib import Path
import albumentations as A

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from yolov5.utils.general import xyxy2xywh

help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(list_path, img_size, batch_size, stride, hyperparameters: dict = None, workers=8,
                      augmentations: list = [], augment: bool = False):
    dataset = LoadImagesAndLabels(list_path, img_size, batch_size,
                                  hyperparameters=hyperparameters,
                                  stride=int(stride),
                                  augment=augment,
                                  augmentations=augmentations)

    batch_size = min(batch_size, len(dataset))
    nb_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nb_workers,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class LoadImagesAndLabels(Dataset):
    def __init__(self, list_path, img_size=640, batch_size=16, hyperparameters: dict = None,
                 image_weights=False,
                 stride=32,
                 augment: bool = False,
                 augmentations: list = []):
        try:
            image_files = []
            p = str(Path(list_path))
            parent = str(Path(p).parent) + os.sep
            if os.path.isfile(p):
                with open(p, 'r') as t:
                    t = t.read().splitlines()
                    image_files += [x.replace('./', parent) if x.startswith('./') else x for x in t]
            else:
                raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in image_files])
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (list_path, e, help_url))

        nb_images = len(self.img_files)
        assert nb_images > 0, 'No images found in %s. See %s' % (list_path, help_url)
        batch_index = np.floor(np.arange(nb_images) / batch_size).astype(np.int)

        self.nb_images = nb_images  # number of images
        self.batch = batch_index  # batch index of image
        self.img_size = img_size
        self.hyperparameters = hyperparameters
        self.image_weights = image_weights
        self.mosaic = self.hyperparameters['augmentation_mosaic']
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.augment = augment
        self.augmentations = augmentations

        # Define labels
        self.label_files = [
            f"{x.replace('images', 'annotations')}.txt"
            if len(x.split(".")) == 1  # If there's no image extension
            else x.replace(x.split(".")[-1], 'txt')  # If there's an image extension
            for x in self.img_files
        ]

        # Check cache
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Cache labels
        nb_missing, nb_found, nb_empty, nb_duplicate = 0, 0, 0, 0
        pbar = enumerate(self.label_files)
        pbar = tqdm(pbar)
        for i, file in pbar:
            l = self.labels[i]  # label
            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file
                assert (l >= 0).all(), 'negative labels: %s' % file
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nb_duplicate += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                self.labels[i] = l
                nb_found += 1  # file found

            else:
                nb_empty += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty

            pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                cache_path, nb_found, nb_missing, nb_empty, nb_duplicate, nb_images)
        if nb_found == 0:
            s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
            print(s)
            raise s

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                labels = []
                image = Image.open(img)
                image.verify()
                shape = exif_size(image)
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                if len(labels) == 0:
                    labels = np.zeros((0, 5), dtype=np.float32)
                x[img] = [labels, shape]
            except Exception as e:
                x[img] = [None, None]
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        if self.mosaic:
            img, labels = load_mosaic(self, index)

        else:
            # Load image
            img, (height, width) = load_image(self, index)

            ##############
            # cv2.imwrite('/tmp/base.png', img)
            ##############

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape)

            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Get the pixel coordinates back
                # This helps to "load" the labels whatever the resizing of the image (as long as it stays proportional)
                labels = x.copy()
                labels[:, 1] = ratio[0] * width * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * height * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * width * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * height * (x[:, 2] + x[:, 4] / 2) + pad[1]

            ##############
            # cv2.imwrite('/tmp/letterbox.png', img)
            # img2 = img.copy()
            # plot_one_box(labels[0][1:], img2, label='test', line_thickness=3)
            # cv2.imwrite('/tmp/letterbox_boxes.png', img2)
            ##############

        if self.augment:
            transform = A.Compose(self.augmentations, bbox_params=A.BboxParams(format='pascal_voc',
                                                                               label_fields=['class_labels']))
            print(labels)
            bboxes = labels[:, 1:] if len(labels) > 0 else []
            class_labels = labels[:, 0] if len(labels) > 0 else []
            transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            if len(labels) > 0:
                labels[:, 1:] = bboxes

            ##############
            # cv2.imwrite('/tmp/image_new.png', img)
            # img2 = img.copy()
            # plot_one_box(labels[0][1:], img2, label='test', line_thickness=3)
            # cv2.imwrite('/tmp/image_new_boxes.png', img2)
            ##############

            # # Augment image space
            # if not self.mosaic:
            #     img, labels = random_perspective(img, labels,
            #                                      degrees=hyp['augmentation_rotation_degrees'],
            #                                      translate=hyp['augmentation_translation_fraction'],
            #                                      scale=hyp['augmentation_scale_gain'],
            #                                      shear=hyp['augmentation_shear_degrees'],
            #                                      perspective=hyp['augmentation_perspective_fraction'])
            #
            # # Augment colorspace
            # augment_hsv(img, hue_gain=hyp['augmentation_hsv_hue'], saturation_gain=hyp['augmentation_hsv_saturation'],
            #             value_gain=hyp['augmentation_hsv_value'])
            #
            # ##############
            # cv2.imwrite('/tmp/hsv.png', img)
            # ##############
            #
            # if random.random() < hyp['augmentation_gaussian_noise_probability']:
            #     gaussian = np.random.normal(0, hyp['augmentation_gaussian_noise_std'], img.shape).astype('uint8')
            #     img = cv2.add(img, gaussian)
            #
            # ##############
            # cv2.imwrite('/tmp/noise.png', img)
            # img2 = img.copy()
            # plot_one_box(labels[0][1:], img2, label='test', line_thickness=3)
            # cv2.imwrite('/tmp/noise_boxes.png', img2)
            # ##############

            # TODO: re-add background augments
            #
            # if random.random() < hyp['augmentation_backgrounds_probability']:
            #     img, labels = augment_background(img, labels)
            #
            # ##############
            # cv2.imwrite('/tmp/background.png', img)
            # img2 = img.copy()
            # plot_one_box(labels[0][1:], img2, label='test', line_thickness=3)
            # cv2.imwrite('/tmp/background_boxes.png', img2)
            # ##############

        nb_labels = len(labels)
        if nb_labels:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        # if self.augment:
        #     # flip up-down
        #     if random.random() < hyp['augmentation_flip_up_down_probability']:
        #         img = np.flipud(img)
        #         if nb_labels:
        #             labels[:, 2] = 1 - labels[:, 2]
        #
        #     # flip left-right
        #     if random.random() < hyp['augmentation_flip_left_right_probability']:
        #         img = np.fliplr(img)
        #         if nb_labels:
        #             labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nb_labels, 6))
        if nb_labels:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index]

    @staticmethod
    def collate_fn(batch):
        img, label, path = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original height width, resized height width
    path = self.img_files[index]
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    height_original, width_original = img.shape[:2]  # orig hw
    r = self.img_size / max(height_original, width_original)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        img = cv2.resize(img, (int(width_original * r), int(height_original * r)),
                         interpolation=all_interpolation[np.random.randint(0, len(all_interpolation))])
    return img, img.shape[:2]  # img, hw_original, hw_resized


def augment_hsv(img, hue_gain=0.5, saturation_gain=0.5, value_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hue_gain, saturation_gain, value_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_mosaic(self, index):
    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyperparameters['augmentation_rotation_degrees'],
                                       translate=self.hyperparameters['augmentation_translation_fraction'],
                                       scale=self.hyperparameters['augmentation_scale_gain'],
                                       shear=self.hyperparameters['augmentation_shear_degrees'],
                                       perspective=self.hyperparameters['augmentation_perspective_fraction'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


all_interpolation = [
    cv2.INTER_NEAREST,
    cv2.INTER_LINEAR,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4
]
padding_color = [114, 114, 114]


def letterbox(img, new_shape: int = 640):
    """
    Resize image to a 32-pixel-multiple rectangle. Keep the proportions by adding constant padding
    :return: the resized image, the ratio tuple (r_height, r_width) where r_height=new_height / old_height, the pad
    tuple (p_vertical, p_horizontal)
    """
    # Make sure new_shape is correct
    assert new_shape / 32 == new_shape // 32

    height, width = img.shape[:2]

    # Scale ratio (new / old)
    ratio = min(new_shape / height, new_shape / width)

    # Compute padding
    new_unpad_width = int(round(width * ratio))
    new_unpad_height = int(round(height * ratio))
    padding_width = (new_shape - new_unpad_width)
    padding_height = (new_shape - new_unpad_height)

    if new_unpad_width != new_shape and new_unpad_width != new_shape:
        # Random resizing technique
        img = cv2.resize(img, (new_unpad_width, new_unpad_height),
                         interpolation=all_interpolation[np.random.randint(0, len(all_interpolation))])

    padding_top = padding_height // 2
    padding_bottom = padding_height - padding_height // 2
    padding_left = padding_width // 2
    padding_right = padding_width - padding_width // 2
    img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT,
                             value=padding_color)
    return img, (ratio, ratio), (padding_left, padding_top)


def augment_background(img, labels):
    # TODO: get randomly one background
    background = cv2.imread('tmp/backgrounds/test.png', cv2.IMREAD_UNCHANGED).astype('uint8')
    labels = labels.astype(np.int)
    img_new = np.copy(background)
    labels_new = np.zeros(labels.shape)

    if img_new.shape != img.shape:
        height, width, _ = img.shape
        img_new = cv2.resize(img_new, (width, height))

    for i, label in enumerate(labels):
        # TODO: make sure objects added one by one can't superpose
        crop = img[label[2]:label[4], label[1]:label[3], :]

        crop_height, crop_width, _ = crop.shape
        img_height, img_width, _ = img_new.shape
        max_x = img_width - crop_width
        max_y = img_height - crop_height
        x1 = np.random.randint(0, max_x)
        y1 = np.random.randint(0, max_y)
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        img_new[y1:y2, x1:x2] = crop
        labels_new[i] = [label[0], x1, y1, x2, y2]

    return img_new, np.array(labels_new).astype(np.float)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates
