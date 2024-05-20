import glob
import os
from pathlib import Path

import torch

from src import preprocessing

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'


def test_preprocessing():
    img_list = (DATA_FILE_PATH / 'images').glob('*.png')
    img_generator = preprocessing.ImageDataGenerator()
    for img in img_list:
        img_tensor, label = img_generator.__getitem__(img)
        assert torch.is_tensor(img_tensor)

