import torch
import os
from torchvision.transforms import v2
import cv2

from src.annotate import Annotate


class ImageDataGenerator(Annotate):
    def __init__(self, transform=None, target_transform=None):
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
                print(self.data_dict)
            else:
                raise TypeError
        elif self.is_images_directory:
            print("Annotations not found. Please click enter to draw annotations manually ...")
            self.process_images(self.path_images)
            self.data_dict, self.img_info_dict = self.get_json_data(self.path_annotations)

        self.img_labels = self.data_dict
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, img_name):
        image = cv2.imread(str(img_name))
        transforms = v2.Compose([
            v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
            v2.Resize(size=(64, 128), antialias=True),  # Or Resize(antialias=True)
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        ])
        img_tensor = transforms(image)

        # print(img_tensor.shape)
        label = self.img_labels[os.path.basename(img_name)]
        img_info = self.img_info_dict[os.path.basename(img_name)]
        return img_tensor, label
