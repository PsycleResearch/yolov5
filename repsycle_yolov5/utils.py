import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from config import image_size, device
import math


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def preprocessing(images):
    # image, _, _ = letterbox(image)
    print(images.shape)
    for image in images:
        image = cv2.resize(image, image_size)
        image = image / 255.
        image = image.reshape((3, image.shape[0], image.shape[1]))
        image = torch.tensor(image).float().to(device)
    return images

def plot_images(image, bboxes, ):
    #image = image * 255.
    H, W, _ = image.shape
    for bboxe in bboxes:
        x, y, w, h, c = bboxe
        p1 = (int((x - w / 2) * W), int((y - h / 2) * H))
        p2 = (int((x + w / 2) * W), int((y + h / 2) * H))
        image = cv2.rectangle(image, p1, p2, color=(255, 0, 0), thickness=3)

    plt.figure(figsize = (20, 20))
    plt.imshow(image)
    plt.show()

def cell_to_coordinates(targets):

    bboxes = []

    for target in targets:

        c_w, c_h = target.shape[1:3]

        for s in range(3):
            for i in range(c_h):
                for j in range(c_w):
                    if target[s, i, j, 4] == 1.:
                        x = ( j + target[s, i, j, 0:1].data.tolist()[0] ) / c_w
                        y = ( i + target[s, i, j, 1:2].data.tolist()[0] ) / c_h
                        w = target[s, i, j, 2:3].data.tolist()[0] / c_w
                        h = target[s, i, j, 3:4].data.tolist()[0] / c_h
                        c = torch.argmax(target[s, i, j, 5:]).data.tolist()
                        bboxes.append([x, y, w, h, c])

    return bboxes

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def pred2bboxes(y, threshold, scaled_anchors):

    predictions = []

    for p in range(3):

        objectness = torch.sigmoid(y[p][..., 4])
        masked_objectness = objectness > threshold
        S = 1 / y[p].shape[-2]

        if True in masked_objectness:

            _, d, cell_i, cell_j = torch.nonzero(masked_objectness, as_tuple=True)

            px = ((cell_j + (y[p][..., 0][masked_objectness].sigmoid() * 2 - 0.5)) * S)
            py = ((cell_i + (y[p][..., 1][masked_objectness].sigmoid() * 2 - 0.5)) * S)
            pw = ((y[p][..., 2][masked_objectness].sigmoid() * 2) ** 2 * scaled_anchors[p, d, 0] * S)
            ph = ((y[p][..., 3][masked_objectness].sigmoid() * 2) ** 2 * scaled_anchors[p, d, 1] * S)
            c = torch.argmax(y[p][..., 5:][masked_objectness], axis=-1)
            o = torch.max(torch.max(y[p][..., 5:][masked_objectness].sigmoid(), axis=-1)[0]) * objectness[masked_objectness]

            predictions.append(torch.stack((px, py, pw, ph, o, c)).T.tolist())

    flatten_pred = []
    for pred_t in predictions:
        for pred in pred_t:
            flatten_pred.append(pred)

    return flatten_pred

def target2bboxes(target, scaled_anchors):
    S = 1 / target.shape[-2]
    cell_i = torch.cumsum(torch.ones(target.shape[:-1]), axis=-2).to(device) - 1
    cell_j = torch.cumsum(torch.ones(target.shape[:-1]), axis=-1).to(device) - 1
    px = (cell_i + target[..., 0].sigmoid()) * S
    py = (cell_j + target[..., 1].sigmoid()) * S
    pw = target[..., 2].exp() * scaled_anchors[:, 0].reshape(3,1,1) * S
    ph = target[..., 3].exp() * scaled_anchors[:, 1].reshape(3,1,1) * S
    objectness = torch.sigmoid(target[..., 4])
    best_class = torch.argmax(target[..., 5:], axis=-1)
    bboxes = torch.stack((px, py, pw, ph, objectness, best_class), dim=-1)
    return bboxes.reshape((4, target.shape[1] * target.shape[2] * target.shape[2], 6)).tolist()

def non_max_suppression(target, scaled_anchors, iou_threshold, threshold):

    from torchvision.ops import nms
    bboxes = pred2bboxes(target, threshold, scaled_anchors)
    print(len(np.asarray(bboxes)))
    if len(bboxes) == 0:
        return []

    bboxes = torch.tensor(bboxes)
    xyxy, scores = bboxes[:,0:4], bboxes[:,4]
    xyxy = xywh2xyxy(xyxy)
    idx = nms(xyxy, scores, iou_threshold)

    return bboxes[idx]

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):

    average_precision = []
    flatten_pred_boxes = []
    flatten_true_boxes = []

    for idx, (key, pred_values) in enumerate(pred_boxes.items()):
        true_values = true_boxes[key]
        for value in pred_values:
            flatten_pred_boxes.append([idx]+value.tolist())
        for value in true_values:
            flatten_true_boxes.append([idx]+value)

    for c in range(num_classes):

        nb_tp = 0
        nb_fp = 0

        precision = []
        recall = []

        max_iou = 0
        max_idx = 0

        used_true_boxes = []
        filtered_true_boxes = [boxes for boxes in flatten_true_boxes if int(boxes[5]) == c]
        filtered_pred_boxes = [boxes for boxes in flatten_pred_boxes if int(boxes[6]) == c]

        sorted_pred_boxes = sorted(filtered_pred_boxes, key=lambda x: x[5], reverse=True)

        for sorted_pred_boxe in sorted_pred_boxes:

            current_true_boxes = [boxe for boxe in filtered_true_boxes if boxe[0] == sorted_pred_boxe[0]]

            if len(current_true_boxes) == 0:
                continue

            for i, true_boxe in enumerate(current_true_boxes):

                current_iou = bbox_iou(torch.tensor(sorted_pred_boxe), torch.tensor(true_boxe), x1y1x2y2=False)

                if current_iou > max_iou:
                    max_iou = current_iou
                    max_idx = i

            if max_iou > iou_threshold and current_true_boxes[max_idx][5] == sorted_pred_boxe[6] \
                    and current_true_boxes[max_idx] not in used_true_boxes:
                nb_tp += 1
                used_true_boxes.append(current_true_boxes[max_idx])
            else:
                nb_fp += 1

            precision.append(nb_tp / (nb_tp + nb_fp))
            recall.append(nb_tp / len(filtered_true_boxes))

        average_precision.append(np.trapz(precision, recall))

        if len(precision) > 0:
            prec = precision[-1]
        else :
            prec = 0

        if len(recall) > 0:
            rec = recall[-1]
        else:
            rec = 0

    return sum(average_precision) / len(average_precision), prec, rec