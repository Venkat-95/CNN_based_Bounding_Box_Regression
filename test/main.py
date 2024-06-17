from pathlib import Path

import torch

from src.preprocessing import Preprocess

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'

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
    img_tensor,_ = dataset.__getitem__(idx=0)
    assert torch.is_tensor(img_tensor)

