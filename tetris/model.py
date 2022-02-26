import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state_curr, action, state_after, reward, game_finished):
        state_curr = torch.tensor(state_curr, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        state_after = torch.tensor(state_after, dtype=torch.float)
        
        if (len(state_curr.shape) == 1):
            state_curr = torch.unsqueeze(state_curr, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            state_after = torch.unsqueeze(state_after, 0)
            game_finished = (game_finished, )

        # predict with q values
        pred = self.model(state_curr) # Qold

        target = pred.clone() # For calculating loss

        for idx in range(len(game_finished)): # for every memory available(?)
            Qnew = reward[idx] # if done, just use the current reward (-10) as Qnew
            if not game_finished[idx]:
                Qnew = reward[idx] + self.gamma * torch.max(self.model(state_after[idx])) # Bellman equation

            target[idx][torch.argmax(action[idx]).item()] = Qnew # Update target with the new Q value to calculate loss

        # calculate loss with nn.MSELoss, optim to clear the previous grad, backward() to calculate grad as loss
        self.optimizer.zero_grad()
        
        # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
        # MSELOSS
        # CLASStorch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')[SOURCE]
        # Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input xx and target yy.
        
        # I still don't understand why the input is "target" and the target is "pred"
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step() # idk what this does