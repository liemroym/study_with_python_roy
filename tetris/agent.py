from collections import deque
from game import Game
from model import Model, QTrainer

MEMORY_SIZE = 100_000
BATCH_SIZE = 1_000

class Agent():
    def __init__(self):
        self.memory = deque()
        self.model = Model(200, 256, 7)
        self.trainer = QTrainer()
        pass

    def get_action(self, state):
        pass

    def train_short(self):
        self.trainer.train_step()

    def train_long(self):
        pass

    def save(self):
        pass

# moves:
    # rotate_cw
    # rotate_ccw
    # soft_drop
    # hard_drop
    # move_left
    # move_right
    # hold

def train():
    agent = Agent()
    game = Game()
    while True:
        curr_state = game.get_state()
        # print(curr_state)
        action = [0, 0, 0, 0, 0, 0, 0]
        # action = agent.get_action(curr_state)
        game.check_move(action)
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