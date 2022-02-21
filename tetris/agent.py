from game import Game

class Agent():
    def __init__():
        pass

    def get_action(self, state):
        pass

    def train_short(self):
        pass

    def train_long(self):
        pass

    def save(self):
        pass

def train():
    agent = Agent()
    game = Game()
    while True:
        curr_state = game.get_state()
        action = agent.get_action(curr_state)
        game.do_it(action)
        reward, game_finished = game.get_state()
        agent.train_short()
        agent.save()

        if (game_finished):
            agent.train_long()
        # getstate
        # take an action
        # do it
        # getState again
        # train short
        # save

if __name__ == "__main__":
    train()