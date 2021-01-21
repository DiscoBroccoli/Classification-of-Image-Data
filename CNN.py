import torch.nn as nn
import torch.nn.functional as F

# Definition of the CNN
# Subclass the torch.nn Module
# Model shape is based on LeNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Note: image input dimensions should be 32 x 32 x 3

        # First convolution, 2D convolution with 3 input channels, 6 output channels (6 filters) and a kernel of size 5x5 (Padding 0, Stride 1)
        # Dimensions should be 32 - 5 + 1 = 28, 28 x 28 x 6
        self.conv1 = nn.Conv2d(3, 6, 5)

        # Max pooling with kernel size of 2 and stride of 2
        # Dimensions should be 14 x 14 x 6
        self.pool = nn.MaxPool2d(2, 2)

        # Second convolution layer, 2D convolution with 6 input channels, 16 output channels (16 filters) and a kernel size of 5x5
        # Dimensions should be 14 - 5 + 1 = 10, 10 x 10 x 6
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully connected layer, 16 * 5 * 5 input features which are the dimension of the input stacked on top, 16 channels, 5x5
        # 120 output features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # Second fully connected Layer, 120 input features, 84 output features
        self.fc2 = nn.Linear(120, 84)

        # Third fully connected layer, 84 input features, 10 output features
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # Apply first convolution layer followed by ReLU activation and Max Pooling
        # Dimensions = 14 x 14 x 6
        x = self.pool(F.relu(self.conv1(x)))

        # Apply second convolution layer followed by ReLU activation and Max Pooling
        # Dimensions = 5 x 5 x 16
        x = self.pool(F.relu(self.conv2(x)))

        # Reshape the tensor into 16 * 5 * 5 columns, flattens the tensor
        x = x.view(-1, 16 * 5 * 5)

        # Apply Fully connected layers 1, 2 and 3 with ReLU activation functions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x