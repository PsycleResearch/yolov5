from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from yolov5.utils.general import (
    non_max_suppression, clip_coords, plot_images, xywh2xyxy, box_iou,
    output_to_target, ap_per_class, print_scores)


def test(model,
         conf_thres,
         iou_thres,
         dataloader,
         save_dir):
    # Initialize/load model and set device
    device = next(model.parameters()).device  # get model device

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    nb_classes = len(model.classes)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    stats = []  # correct, conf, pcls, tcls
    print("VALIDATING")
    for batch_i, (img, targets, paths) in enumerate(tqdm(dataloader)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0
        targets = targets.to(device)
        _, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        with torch.no_grad():
            inf_out, _ = model(img)
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nb_labels = len(labels)
            target_class = labels[:, 0].tolist() if nb_labels else []  # target class
            seen += 1

            if pred is None:
                if nb_labels:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), target_class))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nb_labels:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nb_labels:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

        # Plot images
        if batch_i < 1:
            f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            plot_images(img, targets, paths, str(f), model.classes)  # ground truth
            f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
            plot_images(img, output_to_target(output, width, height), paths, str(f), model.classes)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    p, r, ap, f1, ap_class = ap_per_class(*stats)
    p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    precision = p.mean()
    recall = r.mean()
    map50 = ap50.mean()
    map = ap.mean()
    nb_targets = np.bincount(stats[3].astype(np.int64), minlength=nb_classes)  # number of targets per class

    # Print results
    print_scores('all', seen, nb_targets.sum(), precision, recall, map50, map)

    # Print results per class
    if nb_classes > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print_scores(model.classes[c], seen, nb_targets[c], p[i], r[i], ap50[i], ap[i])

    # Return results
    model.float()  # for training
    maps = np.zeros(nb_classes) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return precision, recall, map50, map
