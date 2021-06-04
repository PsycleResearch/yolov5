import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from config import image_size, device

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    # print(bboxes)

    assert type(bboxes) == list

    # bboxes = [box for box in bboxes if box[4] > threshold

    pred2bboxes()

    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def preprocessing(images):
    # image, _, _ = letterbox(image)
    print(images.shape)
    for image in images:
        image = cv2.resize(image, image_size)
        image = image / 255.
        image = image.reshape((3, image.shape[0], image.shape[1]))
        image = torch.tensor(image).float().to(device)
    return images

def plot_images(image, bboxes):
    #image = image * 255.
    H, W, _ = image.shape
    for bboxe in bboxes:
        x, y, w, h, o, c = bboxe
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

def box_giou(box1, box2):  # box format: (cx, cy, w, h)
    cx1, cy1, w1, h1 = box1.T
    cx2, cy2, w2, h2 = box2.T

    b1_x1, b1_x2 = cx1 - w1 / 2, cx1 + w1 / 2
    b1_y1, b1_y2 = cy1 - h1 / 2, cy1 + h1 / 2
    b2_x1, b2_x2 = cx2 - w2 / 2, cx2 + w2 / 2
    b2_y1, b2_y2 = cy2 - h2 / 2, cy2 + h2 / 2

    ws = torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)
    hs = torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    inter = ws.clamp(min=0) * hs.clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c_area = cw * ch
    return iou - (c_area - union) / c_area

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def pred2bboxes_(y, threshold, scaled_anchors):

    predictions = []

    for p in range(3):

        objectness = torch.sigmoid(y[p][..., 4])
        masked_objectness = objectness > threshold
        S = 1 / y[p].shape[-2]

        if True in masked_objectness:

            _, d, cell_i, cell_j = torch.nonzero(masked_objectness, as_tuple=True)

            px = ((cell_j + y[p][..., 0][masked_objectness].sigmoid()) * S)
            py = ((cell_i + y[p][..., 1][masked_objectness].sigmoid()) * S)
            pw = ((y[p][..., 2][masked_objectness].exp()) * scaled_anchors[p, d, 0] * S)
            ph = ((y[p][..., 3][masked_objectness].exp()) * scaled_anchors[p, d, 1] * S)
            o = objectness[masked_objectness]
            c = (torch.argmax(y[p][..., 5:][masked_objectness]).unsqueeze(dim=-1))
            predictions.append(torch.cat((px, py, pw, ph, o, c)).data.tolist())

    return predictions

def pred2bboxes__(y, threshold, scaled_anchors):

    predictions = []

    for p in range(3):

        objectness = torch.sigmoid(y[p][..., 4])
        masked_objectness = objectness > threshold
        S = 1 / y[p].shape[-2]

        if True in masked_objectness:

            _, d, cell_i, cell_j = torch.nonzero(masked_objectness, as_tuple=True)

            px = ((cell_j + y[p][..., 0][masked_objectness].sigmoid()) * S)
            py = ((cell_i + y[p][..., 1][masked_objectness].sigmoid()) * S)
            pw = ((y[p][..., 2][masked_objectness].exp()) * scaled_anchors[p, d, 0] * S)
            ph = ((y[p][..., 3][masked_objectness].exp()) * scaled_anchors[p, d, 1] * S)
            o = objectness[masked_objectness]
            c = torch.argmax(y[p][..., 5:][masked_objectness], axis=-1)
            predictions.append(torch.stack((px, py, pw, ph, o, c)).T.tolist())

    flatten_pred = []
    for pred_t in predictions:
        for pred in pred_t:
            flatten_pred.append(pred)

    return flatten_pred

def target2bboxes(target, threshold, scaled_anchors):

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

def pred2bboxes(y, scaled_anchors):

    predictions = []

    for p in range(3):

        S = 1 / y[p].shape[-2]
        cell_i = torch.cumsum(torch.ones(y[p].shape[:-1]), axis = 2).to(device)
        cell_j = torch.cumsum(torch.ones(y[p].shape[:-1]), axis = 3).to(device)
        px = torch.flatten((cell_j + y[p][..., 0].sigmoid()) * S)
        py = torch.flatten((cell_i + y[p][..., 1].sigmoid()) * S)
        pw = torch.flatten(y[p][..., 2].exp() * scaled_anchors[p, :, 0].reshape(3,1,1) * S)
        ph = torch.flatten(y[p][..., 3].exp() * scaled_anchors[p, :, 1].reshape(3,1,1) * S)
        objectness = torch.flatten(torch.flatten(torch.sigmoid(y[p][..., 4])))
        c = torch.flatten(torch.argmax(y[p][..., 5:].sigmoid(), axis=-1))
        predictions = predictions + torch.stack((px, py, pw, ph, objectness, c)).T.squeeze(dim=0).tolist()

    return predictions

def non_max_suppression(target, scaled_anchors, iou_threshold, threshold, box_format="midpoint"):

    bboxes = pred2bboxes__(target, threshold, scaled_anchors)
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []

    while bboxes:

        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[5] != chosen_box[5]
            or intersection_over_union(
                torch.tensor(chosen_box[0:4]),
                torch.tensor(box[0:4]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):

    average_precision = []

    flatten_pred_boxes = []
    flatten_true_boxes = []

    for idx, (key, pred_values) in enumerate(pred_boxes.items()):
        true_values = true_boxes[key]
        for value in pred_values:
            flatten_pred_boxes.append([idx]+value)
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

                current_iou = intersection_over_union(torch.tensor(sorted_pred_boxe), torch.tensor(true_boxe))

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

    return sum(average_precision) / len(average_precision)


def mean_average_precision_(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Video explanation of this function:
    https://youtu.be/FppOzcDvaDI
    This function calculates mean average precision (mAP)
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


# def plot_image(image, boxes):
#     """Plots predicted bounding boxes on the image"""
#     cmap = plt.get_cmap("tab20b")
#     class_labels = config.COCO_LABELS if config.DATASET=='COCO' else config.PASCAL_CLASSES
#     colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
#     im = np.array(image)
#     height, width, _ = im.shape
#
#     # Create figure and axes
#     fig, ax = plt.subplots(1)
#     # Display the image
#     ax.imshow(im)
#
#     # box[0] is x midpoint, box[2] is width
#     # box[1] is y midpoint, box[3] is height
#
#     # Create a Rectangle patch
#     for box in boxes:
#         assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
#         class_pred = box[0]
#         box = box[2:]
#         upper_left_x = box[0] - box[2] / 2
#         upper_left_y = box[1] - box[3] / 2
#         rect = patches.Rectangle(
#             (upper_left_x * width, upper_left_y * height),
#             box[2] * width,
#             box[3] * height,
#             linewidth=2,
#             edgecolor=colors[int(class_pred)],
#             facecolor="none",
#         )
#         # Add the patch to the Axes
#         ax.add_patch(rect)
#         plt.text(
#             upper_left_x * width,
#             upper_left_y * height,
#             s=class_labels[int(class_pred)],
#             color="white",
#             verticalalignment="top",
#             bbox={"color": colors[int(class_pred)], "pad": 0},
#         )
#
#     plt.show()


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    threshold,
    box_format="midpoint",
    device="cuda",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()

# def check_class_accuracy(model, loader, threshold):
#     model.eval()
#     tot_class_preds, correct_class = 0, 0
#     tot_noobj, correct_noobj = 0, 0
#     tot_obj, correct_obj = 0, 0
#
#     for idx, (x, y) in enumerate(tqdm(loader)):
#         x = x.to(config.DEVICE)
#         with torch.no_grad():
#             out = model(x)
#
#         for i in range(3):
#             y[i] = y[i].to(config.DEVICE)
#             obj = y[i][..., 0] == 1 # in paper this is Iobj_i
#             noobj = y[i][..., 0] == 0  # in paper this is Iobj_i
#
#             correct_class += torch.sum(
#                 torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
#             )
#             tot_class_preds += torch.sum(obj)
#
#             obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
#             correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
#             tot_obj += torch.sum(obj)
#             correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
#             tot_noobj += torch.sum(noobj)
#
#     print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
#     print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
#     print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
#     model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_loaders(train_csv_path, test_csv_path):
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader

def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes)



def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False