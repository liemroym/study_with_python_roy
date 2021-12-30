from game_ori import Game

MAX_MEMORY = 100000 # max memory (using deque, remove oldest memory)
BATCH_SIZE = 1000
LR = 0.001 # learning rate, used in Bell equation

class Agent:
    def __init__(self):
        self.epsilon = 0 # randomness, used for early random moves
        self.gamma = 0 # discount rate, used in Bell equation

    def getState(self, game):
        pass

    def save(self, currState, reward, score, newState, gameOver):
        pass

    # train for 1 step (1 currState, newState, reward, gameOver, score)
    def trainShort(self):
        pass
    
    # train for 1 batch (BATCH_NUMBER currState, ...)
    def trainLong(self):
        pass

    def getAction(self, state):
        pass

# training function that gets called
def train():
    plotScores = []
    plotMeanScores = []
    record = 0

    game = Game()
    agent = Agent()

    while True:
        currState = agent.getState(game)
        move = agent.getAction(currState)
        reward, gameOver, score = game.moveSnake(move)
        newState = agent.getState(game)
        agent.save(currState, reward, score, newState, gameOver)
        agent.trainShort()

        if gameOver:
            agent.trainLong()
if __name__ == '__main__':
    train()