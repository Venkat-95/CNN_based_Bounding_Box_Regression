import cv2
import torch


def normalize_bounding_boxes(bboxes, original_size):
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
    orig_h, orig_w = original_size

    normalized_bboxes = bboxes.clone()
    normalized_bboxes[:, [0, 2]] /= orig_w
    normalized_bboxes[:, [1, 3]] /= orig_h

    return normalized_bboxes


def denormalize_bounding_boxes(bboxes, original_size):
    orig_w, orig_h = original_size
    denormalized_bboxes = bboxes.clone()
    denormalized_bboxes[:, [0, 2]] *= orig_w
    denormalized_bboxes[:, [1, 3]] *= orig_h
    return denormalized_bboxes


def draw_bounding_boxes(image, bboxes, color=(255, 0, 0), thickness=2):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image


def compute_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection
    _iou = intersection / union if union > 0 else 0

    return _iou


