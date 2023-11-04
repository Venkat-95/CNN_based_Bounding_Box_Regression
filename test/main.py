import glob
import os

import torch

from CNN_based_Bounding_Box_Regression.src import preprocessing



def test_preprocessing():
    for i in glob.glob("../data/images/*.png"):
        img_generator = preprocessing.ImageDataGenerator("../data/images")
        img_tensor, label = img_generator.__getitem__(os.path.basename(i))
        assert torch.is_tensor(img_tensor)