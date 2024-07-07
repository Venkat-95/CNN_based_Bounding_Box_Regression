from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.model import customNN
from src.preprocessing import Preprocess

CURRENT_FILE_PATH = Path(__file__).resolve()
DATA_FILE_PATH = CURRENT_FILE_PATH.parent.parent / 'data'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

dataset = Preprocess(device=device)
train_dataset = torch.utils.data.TensorDataset(dataset.X_train, dataset.Y_train)
test_dataset = torch.utils.data.TensorDataset(dataset.X_test, dataset.Y_test)
image_names_train = dataset.image_name_train
image_names_test = dataset.image_name_test
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
model = customNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion_bbox = nn.MSELoss()
criterion_score = nn.BCELoss()

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs_bbox, outputs_score = model(images)
        true_scores = torch.ones_like(outputs_score, device=device)

        loss_bbox = criterion_bbox(outputs_bbox, targets)
        loss_score = criterion_score(outputs_score, true_scores)

        loss = loss_bbox + loss_score

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}')

print('Finished Training')
torch.save(model,DATA_FILE_PATH/'trained_model.pth')

