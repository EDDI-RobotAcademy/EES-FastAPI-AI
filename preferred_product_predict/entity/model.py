import torch.nn as nn


class PreferredProductPredictModel(nn.Module):
    def __init__(self, in_features=2, out_features=47, hidden_size=64):
        super(PreferredProductPredictModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, out_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
