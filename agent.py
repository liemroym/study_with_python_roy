# followed series: https://www.youtube.com/watch?v=VGkcmBaeAGM

from numpy.random.mtrand import rand
import torch
import random
import numpy as np
from game import Game
from collections import deque

MAX_MEMORY = 100000 # max memory (using deque, remove oldest memory)
BATCH_SIZE = 1000
LR = 0.001 # learning rate, used in Bell equation

class Agent:
    def __init__(self):
        self.ngames = 0 # for tracking how many games iterated, deciding when randomness disappear
        self.epsilon = 0 # randomness, used for early random moves
        self.gamma = 0 # discount rate, used in Bell equation
        self.memory = deque(maxlen=MAX_MEMORY) # data structure, for saving memories (currstate, action, etc)
        self.model = None
        self.trainer = None

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
            (game.y > game.food[0]), # food on up
            (game.y > game.food[0])  # food on down
        ]

        return np.array(state, dtype=int)

    def save(self, currState, action, reward, newState, gameOver):
        self.memory.append((currState, action, reward, newState, gameOver))

    # train for 1 step (1 currState, action, newState, reward, gameOver)
    def trainShort(self, currState, action, reward, newState, gameOver):
        self.trainer.trainStep(currState, action, reward, newState, gameOver)
    
    # train for 1 batch (BATCH_NUMBER currState, ...)
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
            move[random.randint(0, 2)] = 1
        else:
            stateTensor = torch.tensor(state, float)
            prediction = self.model(stateTensor)
            # torch.argmax(tensor).item() --> returns index of maximum argument ([5.03, 4.1, 0.23] --> 0)
            move[torch.argmax(prediction).item()] = 1

        return move

# training function that gets called
def train():
    plotScores = []
    plotMeanScores = []
    record = 0

    game = Game()
    agent = Agent()

    while True:
        # do move for current state
        currState = agent.getState(game)
        move = agent.getAction(currState)
        reward, gameOver, score = game.startGame(move)
        newState = agent.getState(game)
        agent.trainShort(currState, move, reward, newState, gameOver)

        agent.save(currState, reward, score, newState, gameOver)

        if gameOver:
            # train long + plot
            game.reset()
            agent.ngames += 1
            agent.trainLong(currState, move, reward, newState, gameOver)

            if (score > record):
                record = score



if __name__ == '__main__':
    train()