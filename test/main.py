import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader

from src.preprocessing import Preprocess
from src.utils import denormalize_bounding_boxes, draw_bounding_boxes, compute_iou
from src.model import customNN

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'
SCORE_THRESHOLD = 0.80
IOU_THRESHOLD = 0.5
VISUALIZE_BOUNDING_BOX = False

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


def test_preprocessing_is_valid_tensor():
    dataset = Preprocess(device=DEVICE)
    img_tensor, _ = dataset.__getitem__(idx=0)
    assert torch.is_tensor(img_tensor)


def test_trained_model():
    _model_cnn = customNN()
    model = torch.load(DATA_FILE_PATH / 'trained_model.pth', map_location=DEVICE)
    model.eval()
    predictions = {}
    prediction_score = {}
    ground_truth = {}
    dataset = Preprocess(device=DEVICE)
    test_dataset = torch.utils.data.TensorDataset(dataset.X_test, dataset.Y_test)
    image_names_test = dataset.image_name_test
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for index, (images, targets) in enumerate(test_dataloader):
            img_name = image_names_test[index]
            outputs_bbox, outputs_score = model(images)
            ground_truth[img_name] = targets.cpu().numpy()
            for i in range(outputs_bbox.size(0)):
                prediction_score[img_name] = outputs_score[i].cpu().numpy()
                if prediction_score[img_name] > SCORE_THRESHOLD:
                    predictions[img_name] = outputs_bbox[i].cpu().numpy()

    for img_name, prediction in predictions.items():
        print(f'Image: {img_name}, Prediction: {prediction}')
        image = cv2.imread(os.path.join(DATA_FILE_PATH, "images", img_name))
        image_size = image.shape[1], image.shape[0]

        prediction_tensor = torch.tensor(prediction).unsqueeze(0)
        denormalized_bboxes = denormalize_bounding_boxes(prediction_tensor, image_size).numpy()

        image_with_bboxes = draw_bounding_boxes(image, denormalized_bboxes)

        if VISUALIZE_BOUNDING_BOX:
            cv2.imshow(f'Image: {img_name}', image_with_bboxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    for img_name, pred_bbox in ground_truth.items():
        iou_scores = compute_iou(predictions[img_name], ground_truth[img_name][0])
        print(iou_scores)
