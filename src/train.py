import numpy as np
import cv2
import torch
from torch import nn
from src.preprocessing import ImageDataGenerator


class preprocess():
    def __init__(self):
        dataGenerator = ImageDataGenerator()

        self.X_train = torch.empty(3,64, 128)
        self.Y_train = torch.empty(4)
        self.X_test = torch.empty(3,64, 128)
        self.Y_test = torch.empty(4)

        select_indices_test = list(np.random.randint(0, len(dataGenerator.data_dict), int(0.1 * len(dataGenerator.data_dict))))
        test_indices = []
        for i in range(0,len(select_indices_test)):
            test_indices.append([*dataGenerator.data_dict][i])
        count_train = 0
        count_test = 0
        for key,value in dataGenerator.data_dict.items():

            x1 = dataGenerator.data_dict[key][0][1][0] * (128/dataGenerator.img_info_dict[key][0])
            y1 = dataGenerator.data_dict[key][0][1][1] * (64/dataGenerator.img_info_dict[key][1])
            x2 = dataGenerator.data_dict[key][0][1][2] * (128/dataGenerator.img_info_dict[key][0])
            y2 = dataGenerator.data_dict[key][0][1][3] * (64/dataGenerator.img_info_dict[key][1])

            img_tensor, label = dataGenerator.__getitem__(key)

            if key not in test_indices:
                count_train += 1
                self.X_train = torch.cat((img_tensor,self.X_train),2)
                self.Y_train = torch.cat((torch.tensor([x1,y1,x2,y2]),self.Y_train),0)
            else:
                count_test += 1
                self.X_test = torch.cat((img_tensor,self.X_test),2)
                self.Y_test = torch.cat((torch.tensor([x1,y1,x2,y2]),self.Y_test),0)
        self.X_train = self.X_train[:10]
        self.Y_train = self.Y_train[:10]
        self.X_train = self.X_train.to(device)
        self.Y_train = self.Y_train.to(device)
        self.X_test = self.X_test.to(device)
        self.Y_test = self.Y_test.to(device)

        print(self.X_test.size(),self.Y_test.size())
        print(self.X_train.size(),self.Y_test.size())


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            #nn.Linear(100, 100),
            nn.Conv2d(in_channels=3,out_channels=100,kernel_size=3),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,4),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    build_train = preprocess()
    model = NeuralNetwork().to(device)
    print(model)

    logits = model(build_train.X_train).to(device)
    pred_probab = nn.Softmax(dim=1)(logits).to(device)
    y_pred = pred_probab.argmax(1).to(device)
    print(f"Predicted class: {y_pred}")



