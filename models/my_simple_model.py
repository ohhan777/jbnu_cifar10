import torch.nn as nn
import torch.nn.functional as F

class MySimpleModel(nn.Module):
    def __init__(self):
        super(MySimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm128 = nn.BatchNorm2d(128)
        self.linear1 = nn.Linear(in_features=8*8*128, out_features=4096)
        self.linear2 = nn.Linear(4096, 10)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = F.relu(self.batch_norm64(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.batch_norm128(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
