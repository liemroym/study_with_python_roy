import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

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
        # make tensors
        currState = torch.tensor(currState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        newState = torch.tensor(newState, dtype=torch.float)

        # resize to make sure that train short and long (single and batches) can use this function
        if (len(currState.shape) == 1):
            currState = torch.unsqueeze(currState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            newState = torch.unsqueeze(newState, 0)
            gameOver = (gameOver, )

        # predict with q values
        pred = self.model(currState) # Qold

        target = pred.clone()

        for idx in range(len(gameOver)): # for every memory available(?)
            Qnew = reward[idx] # if done, just use the current reward as Qnew
            if not gameOver[idx]:
                Qnew = reward[idx] + self.gamma * torch.max(self.model(newState[idx])) # Bellman equation

            target[idx][torch.argmax(action[idx]).item()] = Qnew # whatever the f this is

        # calculate loss with nn.MSELoss, optim to clear the previous grad, backward() to calculate grad as loss
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        
        loss.backward()

        self.optimizer.step() # idk what this does