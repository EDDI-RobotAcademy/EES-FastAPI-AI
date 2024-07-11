import torch.nn as nn


class UserWithdrawPredictModel(nn.Module):
    def __init__(self, in_features=12, out_features=1, hidden_size=64):
        super(UserWithdrawPredictModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
