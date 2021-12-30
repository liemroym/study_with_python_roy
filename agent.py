from game import Game

MAX_MEMORY = 100000 # max memory (using deque, remove oldest memory)
BATCH_SIZE = 1000
LR = 0.001 # learning rate, used in Bell equation

class Agent:
    def __init__(self):
        self.ngames = 0 # for tracking how many games iterated, deciding when randomness disappear
        self.epsilon = 0 # randomness, used for early random moves
        self.gamma = 0 # discount rate, used in Bell equation

    def getState(self, game):
        pass

    def save(self, currState, action, reward, newState, gameOver):
        pass

    # train for 1 step (1 currState, action, newState, reward, gameOver)
    def trainShort(self, currState, action, reward, newState, gameOver):
        pass
    
    # train for 1 batch (BATCH_NUMBER currState, ...)
    def trainLong(self, currState, action, reward, newState, gameOver):
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
        # do move for current state
        currState = agent.getState(game)
        move = agent.getAction(currState)
        reward, gameOver, score = game.moveSnake(move)
        newState = agent.getState(game)
        agent.trainShort(currState, move, reward, newState, gameOver)

        agent.save(currState, reward, score, newState, gameOver)
        
        if gameOver:
            # train long + plot
            game.reset()
            agent.trainLong(currState, move, reward, newState, gameOver)


if __name__ == '__main__':
    train()