from pathlib import Path

import numpy as np
import torch
import os

from torch.utils.data import Dataset
from torchvision.transforms import v2
import cv2

from src.annotate import Annotate

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'


def normalize_bounding_boxes(bboxes, original_size):
    """
    Normalizes bounding box coordinates to the range [0, 1] based on the original image size.

    Args:
    - bboxes (list or tensor): List or tensor of shape (N, 4) where N is the number of bounding boxes,
      and each bounding box is represented as [x_min, y_min, x_max, y_max].
    - original_size (tuple): The original size of the image as (height, width).

    Returns:
    - normalized_bboxes (tensor): The normalized bounding box coordinates.
    """
    # Convert bboxes to tensor if they are a list of lists
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

    # Extract original dimensions
    orig_h, orig_w = original_size

    # Normalize bounding boxes
    normalized_bboxes = bboxes.clone()
    normalized_bboxes[:, [0, 2]] /= orig_w  # Normalize x coordinates
    normalized_bboxes[:, [1, 3]] /= orig_h  # Normalize y coordinates

    return normalized_bboxes


class ImageDataGenerator(Annotate):
    def __init__(self, transform=None, target_transform=None):
        super().__init__()
        print(f"Returned Responses for Annotations present: {self.annotations_present}")
        if self.annotations_present:
            img_test = os.listdir(self.path_annotations)[0]
            if img_test.endswith(".xml"):
                print("Found Annotations as XML files. Extracting the information to dictionaries ...")
                self.data_dict, self.img_info_dict = self.get_xml_data(self.path_annotations)
            elif img_test.endswith(".json"):
                print("Found Annotations as JSON file. Extracting the information to dictionaries ...")
                self.data_dict, self.img_info_dict = self.get_json_data(self.path_annotations)
            else:
                raise TypeError
        elif self.is_images_directory:
            print("Annotations not found. Please click enter to draw annotations manually ...")
            self.process_images(self.path_images)
            self.data_dict, self.img_info_dict = self.get_json_data(self.path_annotations)
        else:
            raise "No Images / Annotations found in the ./data directory."
        self.img_labels = self.data_dict
        self.transform = transform
        self.target_transform = target_transform


class Preprocess(Annotate, Dataset):
    def __init__(self, device):
        super().__init__()
        print(f"Returned Responses for Annotations present: {self.annotations_present}")
        if self.annotations_present:
            img_test = os.listdir(self.path_annotations)[0]
            if img_test.endswith(".xml"):
                print("Found Annotations as XML files. Extracting the information to dictionaries ...")
                self.data_dict, self.img_info_dict = self.get_xml_data(self.path_annotations)
                print(self.data_dict)
            elif img_test.endswith(".json"):
                print("Found Annotations as JSON file. Extracting the information to dictionaries ...")
                self.data_dict, self.img_info_dict = self.get_json_data(self.path_annotations)
            else:
                raise TypeError
        elif self.is_images_directory:
            print("Annotations not found. Please click enter to draw annotations manually ...")
            self.process_images(self.path_images)
            self.data_dict, self.img_info_dict = self.get_json_data(self.path_annotations)
        else:
            raise "No Images / Annotations found in the ./data directory."
        self.img_labels = self.data_dict


        self.device = device
        self.X_train = torch.empty(3, 64, 128)
        self.Y_train = torch.empty(1, 4)
        self.X_test = torch.empty(3, 64, 128)
        self.Y_test = torch.empty(1, 4)

        select_indices_test = list(
            np.random.randint(0, len(self.data_dict), int(0.1 * len(self.data_dict))))
        test_indices = []
        for i in range(0, len(select_indices_test)):
            test_indices.append([*self.data_dict][i])
        count_train = 0
        count_test = 0
        for i, (key, value) in enumerate(self.data_dict.items()):
            img_tensor, bounding_box_tensor = self.__getitem__(i)

            if key not in test_indices:
                count_train += 1
                self.X_train = torch.cat((img_tensor, self.X_train), 2)
                self.Y_train = torch.cat((bounding_box_tensor, self.Y_train), 0)
            else:
                count_test += 1
                self.X_test = torch.cat((img_tensor, self.X_test), 2)
                self.Y_test = torch.cat((bounding_box_tensor, self.Y_test), 0)
        self.X_train = self.X_train[:10]
        self.Y_train = self.Y_train[:10]
        self.X_train = self.X_train.to(self.device)
        self.Y_train = self.Y_train.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.Y_test = self.Y_test.to(self.device)

        print(f"Shape of the Test Tensor Image Pixels: {self.X_test.size()}")
        print(f"Shape of the Training Tensor Image Pixels: {self.X_train.size()}")
        print(f"Shape of the Training Labels : {self.Y_train.size()}")
        print(f"Shape of the Testing Labels :{self.Y_test.size()}")

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, _ = list(self.data_dict.items())[idx]
        image = cv2.imread(os.path.join(DATA_FILE_PATH / 'images', img_name))
        image_size = image.shape[0:2]
        # image = read_image(os.path.join(DATA_FILE_PATH / 'images', img_name))
        transforms = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.Resize(size=(64, 128), antialias=True),  # Or Resize(antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        ])
        img_tensor = transforms(image)

        bboxes = self.img_labels[os.path.basename(img_name)][0][1]

        if isinstance(bboxes, list):
            bboxes = torch.tensor(bboxes, dtype=torch.float32)

        if bboxes.dim() == 1:  # Single bounding box case
            bboxes = bboxes.unsqueeze(0)
        normalized_bboxes = normalize_bounding_boxes(bboxes, image_size)

        img_info = self.img_info_dict[os.path.basename(img_name)]
        return img_tensor, normalized_bboxes
