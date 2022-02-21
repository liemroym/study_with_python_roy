# followed series: https://www.youtube.com/watch?v=VGkcmBaeAGM

# Tensors can run on GPUs
from numpy.random.mtrand import rand
import torch
import random
import numpy as np
from game import Game
from model import LinearQNet, QTrainer
from plotter import plot
from collections import deque


MAX_MEMORY = 100000 # max memory (using deque, remove oldest memory)
BATCH_SIZE = 1000
LR = 0.01 # learning rate, used in Bell equation

class Agent:
    def __init__(self):
        self.ngames = 0 # for tracking how many games iterated, deciding when randomness disappear
        self.epsilon = 0 # randomness, used for early random moves
        self.gamma = 0.9 # discount rate, used in Bell equation
        self.memory = deque(maxlen=MAX_MEMORY) # data structure, for saving memories (currstate, action, etc)
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def getState(self, game):
        # state:
        # [danger_straight, danger_right, danger_left
        #  direction_left, direction_right, direction_up, direction_down,
        #  food_left, food right, food_up, food_down
        # ]
        # boolean, represented with 0s and 1s

        # check danger with checkCollision
        head = game.snake[0]
        headUp = (head[0], head[1]-20)
        headDown = (head[0], head[1]+20)
        headLeft = (head[0]-20, head[1])
        headRight = (head[0]+20, head[1])

        state = [
            # CHECK DANGER
            # normal
            (game.direction == 'UP' and game.checkCollision(headUp)) or
            (game.direction == 'DOWN' and game.checkCollision(headDown)) or
            (game.direction == 'LEFT' and game.checkCollision(headLeft)) or
            (game.direction == 'RIGHT' and game.checkCollision(headRight)),

            # up --> right --> down --> left (danger right)
            (game.direction == 'UP' and game.checkCollision(headRight)) or
            (game.direction == 'DOWN' and game.checkCollision(headLeft)) or
            (game.direction == 'LEFT' and game.checkCollision(headUp)) or
            (game.direction == 'RIGHT' and game.checkCollision(headDown)),

            # up --> left --> down --> right (danger left)
            (game.direction == 'UP' and game.checkCollision(headLeft)) or
            (game.direction == 'DOWN' and game.checkCollision(headRight)) or
            (game.direction == 'LEFT' and game.checkCollision(headDown)) or
            (game.direction == 'RIGHT' and game.checkCollision(headUp)),

            # CHECK DIRECTION
            # left, right, up, down
            (game.direction == 'LEFT'),
            (game.direction == 'RIGHT'),
            (game.direction == 'UP'),
            (game.direction == 'DOWN'),

            # CHECK FOOD
            # left, right, up, down
            (game.x > game.food[0]), # food on right
            (game.x < game.food[0]), # food on left
            (game.y < game.food[1]), # food on up
            (game.y > game.food[1])  # food on down
        ]

        return np.array(state, dtype=int)

    def save(self, currState, action, reward, newState, gameOver):
        self.memory.append((currState, action, reward, newState, gameOver))

    # train for 1 step (1 currState, action, newState, reward, gameOver)
    def trainShort(self, currState, action, reward, newState, gameOver):
        self.trainer.trainStep(currState, action, reward, newState, gameOver)
    
    # train for 1 batch (BATCH_NUMBER currState, ...)
    # link about zip(*list)
    # https://www.google.com/search?q=zip(*list)+python&rlz=1C1GCEB_enID966ID966&oq=zip(&aqs=chrome.2.69i57j0i512j0i20i263i512j0i512l4j69i61.3942j0j7&sourceid=chrome&ie=UTF-8
    def trainLong(self):
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory
        
        currStates, actions, rewards, newStates, gameOvers = zip(*sample)
        self.trainer.trainStep(currStates, actions, rewards, newStates, gameOvers)

    def getAction(self, state):
        self.epsilon = 80 - self.ngames
        move = [0, 0, 0]
        if (random.randint(0, 200) < self.epsilon):
            predMove = random.randint(0, 2)
            move[predMove] = 1
        else:
            stateTensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(stateTensor)
            # torch.argmax(tensor).item() --> returns index of maximum argument ([5.03, 4.1, 0.23] --> 0) (.item() != index)
            predMove = torch.argmax(prediction).item()
            move[predMove] = 1

        return move

# training function that gets called
def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0 # to count mean
    record = 0

    game = Game()
    agent = Agent()
    
    while True:
        # do move for current state
        currState = agent.getState(game)
        move = agent.getAction(currState)

        # if (move == [1, 0, 0]):
        #     print("STRAIGHT") 
        # elif (move == [0, 1, 0]):
        #     print("RIGHT")
        # else:
        #     print("LEFT")

        # get state --> get move prediction 

        reward, gameOver, score = game.startGame(move)
        newState = agent.getState(game)
        agent.trainShort(currState, move, reward, newState, gameOver)

        agent.save(currState, move, reward, newState, gameOver)

        if gameOver:
            # train long + plot
            game.reset()
            agent.ngames += 1
            agent.trainLong()

            if (score > record):
                record = score
                agent.model.save()

            print('Game:', agent.ngames, 'Score:', score, 'Record:', record)
            plotScores.append(score)
            totalScore += score
            plotMeanScores.append(totalScore / agent.ngames)

            plot(plotScores, plotMeanScores)


if __name__ == '__main__':
    train()