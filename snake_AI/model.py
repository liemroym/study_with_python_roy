import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time

class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        # print("1", x)
        x = F.relu(self.linear1(x))
        # print("2", x)
        x = self.linear2(x)
        # print("3", x)

        return x
    
    def save(self, fileName='model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName)

    
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def trainStep(self, currState, action, reward, newState, gameOver):
        # make tensors (the tensors are filled with tuples or )
        currState = torch.tensor(currState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        newState = torch.tensor(newState, dtype=torch.float)
          
        if (len(currState.shape) == 1):
            currState = torch.unsqueeze(currState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            newState = torch.unsqueeze(newState, 0)
            gameOver = (gameOver, )

        # predict with q values
        pred = self.model(currState) # Qold

        target = pred.clone() # For calculating loss

        for idx in range(len(gameOver)): # for every memory available(?)
            Qnew = reward[idx] # if done, just use the current reward (-10) as Qnew
            if not gameOver[idx]:
                Qnew = reward[idx] + self.gamma * torch.max(self.model(newState[idx])) # Bellman equation

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