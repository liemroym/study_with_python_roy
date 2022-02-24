from collections import deque
import random
from time import sleep
import numpy as np
import torch

from game import Game, GAME_BOARD_POSITION, GRID_SIZE
from model import Model, QTrainer

MEMORY_SIZE = 100_000
BATCH_SIZE = 1_000
N_OF_RANDOMNESS = 80

# actions = [
# 'rotate_cw',
# 'rotate_ccw',
# 'soft_drop',
# 'hard_drop',
# 'move_left',
# 'move_right',
# 'hold',
# ]

class Agent():
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.ngames = 0 # for tracking how many games iterated, deciding when randomness disappear
        self.epsilon = 0 # randomness, used for early random moves
        self.gamma = 0.9 # discount rate, used in Bell equation
        
        # input: 10 * 20 minoes (board size), 4 minoes of current_tetromino, 4 minoes of hold_tetromino, considering next tetrominoes
        # output:
            # 0. rotate_cw
            # 1. rotate_ccw
            # 2. soft_drop
            # 3. hard_drop
            # 4. move_left
            # 5. move_right
            # 6. hold
        self.model : Model = Model(208, 256, 7)
        self.trainer : QTrainer = QTrainer()
        self.count = 0
        pass

    def get_state(self, game : Game):
        coords = np.zeros((20, 10))

        for tetromino in game.tetrominoes:
            for coord in tetromino.coords:
                coords[(coord[1] // 20)- 1][(coord[0] - GAME_BOARD_POSITION[0]) // 20] = 1        

        return coords

    def get_action(self, state):
        self.epsilon = N_OF_RANDOMNESS - self.ngames
        action = np.zeros(7)
        if (random.randrange(0, N_OF_RANDOMNESS) < self.epsilon):
            action[random.randint(0, len(action)-1)] = 1
        else:
            state_torch = torch.tensor(state, dtype=torch.float)
            self.model.forward(state_torch)
            action[torch.argmax(state_torch).item()] = 1

        return action

    def train_short(self):
        self.trainer.train_step()

    def train_long(self):
        pass

    def save(self):
        pass



def train():
    game = Game()
    agent = Agent()
    while True:
        curr_state = agent.get_state(game)
        action = agent.get_action(curr_state)
        # action = [0, 0, 0, 0, 1, 0, 0]
        game.start(action)
        # reward, game_finished = game.get_state()
        # agent.train_short()
        # agent.save()

        # if (game_finished):
        #     agent.train_long()
        # getstate
        # take an action
        # do it
        # getState again
        # train short
        # save

if __name__ == "__main__":
    train()