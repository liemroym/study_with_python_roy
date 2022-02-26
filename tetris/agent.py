from collections import deque
import random
import numpy as np
from time import sleep
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
        self.trainer : QTrainer = QTrainer(self.model, 0.01, 0.9)
        self.count = 0

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

    def train_short(self, state_curr, action, state_after, reward, game_finished):
        self.trainer.train_step(state_curr, action, state_after, reward, game_finished)

    def train_long(self):   
        sample = random.sample(list(self.memory), BATCH_SIZE)

        state_currs, actions, state_afters, rewards, game_finisheds = zip(*sample)
        self.trainer.train_step(state_currs, actions, state_afters, rewards, game_finisheds)

    def save(self, state_curr, action, state_after, reward, game_finished, score):
        self.memory.append((state_curr, action, state_after, reward, game_finished, score))

def train():
    game = Game()
    agent = Agent()
    while True:
        state_curr = game.get_state()
        action = agent.get_action(state_curr)
        # action = [0, 0, 0, 0, 1, 0, 0]
        reward, game_finished, score = game.start(action)
        state_after = game.get_state()
        agent.train_short(state_curr, action, state_after, reward, game_finished)
        agent.save(state_curr, action, state_after, reward, game_finished, score)
        if (game_finished):
            game.reset()
        #  game.get_state()
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