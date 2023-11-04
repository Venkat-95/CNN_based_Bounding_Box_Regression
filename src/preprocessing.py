import glob
import torch
import os
from torchvision.io import read_image, ImageReadMode
from CNN_based_Bounding_Box_Regression.src.annotate import Annotate
from torchvision import transforms as tf
import cv2

class ImageDataGenerator(Annotate):
    def __init__(self, img_dir, transform=None, target_transform=None):
        super().__init__()
        self.img_dir = img_dir

        if self.annotations_present:
            img_test = os.listdir(self.path_annotations)[0]
            if img_test.endswith(".xml"):
                self.data_dict, self.img_info_dict = self.get_xml_data(self.path_annotations)
            elif img_test.endswith(".json"):
                self.data_dict, self.img_info_dict = self.get_json_data(self.path_annotations)
            else:
                raise TypeError
        elif self.is_images_directory:
            self.process_images(self.path_images)
        self.img_labels = self.data_dict
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, img_name):
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.tensor(image)
        label = self.img_labels[img_name]
        print(img_tensor.size())
        return img_tensor, label


if __name__ == "__main__":
    test = ImageDataGenerator("../data/images")
    for i in glob.glob("../data/images/*.png"):
        test.__getitem__(os.path.basename(i))

