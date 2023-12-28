import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        out = self.relu(self.pool(self.conv1(x)))
        out = self.relu(self.pool(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = out.view(x.size(0), -1)
        out = self.fc2(self.relu(self.fc1(out)))
        return self.logsoftmax(out)