import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer)
        self.layer2 = nn.Linear(hidden_layer, output_size)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)

        return x

class QTrainer():
    def __init__(self):
        pass

    def train_step(self):
        pass