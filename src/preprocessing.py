from pathlib import Path

import numpy as np
import torch
import os

from torch.utils.data import Dataset
from torchvision.transforms import v2
import cv2

from src.annotate import Annotate
from src.utils import normalize_bounding_boxes

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'


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
        self.train_images = []
        self.train_bboxes = []
        self.test_images = []
        self.test_bboxes = []
        self.image_name_train = []
        self.image_name_test = []

        for i, (key, value) in enumerate(self.data_dict.items()):
            img_tensor, bounding_box_tensor, image_name = self.__getitem__(i)

            if key not in test_indices:
                count_train += 1
                self.train_images.append(img_tensor)
                self.train_bboxes.append(bounding_box_tensor)
                self.image_name_train.append(image_name)
            else:
                count_test += 1
                self.test_images.append(img_tensor)
                self.test_bboxes.append(bounding_box_tensor)
                self.image_name_test.append(image_name)

        self.X_train = torch.stack(self.train_images)
        self.Y_train = torch.cat(self.train_bboxes)
        self.X_test = torch.stack(self.test_images)
        self.Y_test = torch.cat(self.test_bboxes)

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
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(64, 128), antialias=True),
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
        return img_tensor, normalized_bboxes, img_name
