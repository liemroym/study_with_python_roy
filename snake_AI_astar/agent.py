# Heuristic: Manhattan Distance
from game_astar import *

around = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Agent:
    def __init__(self):
        self.game : Game = Game()

    def find_path(self, state) -> str: 
        snake = state[1]
        food = state[2]

        to_visit = [(abs(food[0]-snake[0][0]) + abs(food[1]-snake[0][1]), snake[0], (-1, -1), 0)] # Cost, node, parent, path cost
        history = []

        # Traverse to find goal
        while (not(len(to_visit) == 0)):
            node = to_visit[0]
            history.append(node)
            if node[1] == food: break

            for dir in around:
                curr_X = node[1][0] + dir[0]
                curr_Y = node[1][1] + dir[1]
                if (curr_X > 0 and curr_X < SCREEN_SIZE[0] // GRID_SIZE):
                    if (curr_Y > 0 and curr_Y < SCREEN_SIZE[1] // GRID_SIZE):
                        if (not ((curr_X, curr_Y) in snake or (curr_X, curr_Y) in list(n[1] for n in history) or (curr_X, curr_Y) in list(n[1] for n in to_visit))):
                            cost = abs(curr_X - food[0]) + abs(curr_Y - food[1]) + node[3]
                            to_visit.append((cost, (curr_X, curr_Y), node[1], node[3]+1))
                
                if ((curr_X, curr_Y) == food): break
            
            to_visit = to_visit[1:]
            to_visit.sort(key=lambda t: t[0])

        # Find direction
        curr_node = history[-1]
        for node in reversed(history):
            if (node != history[0]):
                if (curr_node[2] == node[1]):
                    curr_node = node

        if (curr_node[2][0]-curr_node[1][0] == -1):
            return "RIGHT"
        elif (curr_node[2][0]-curr_node[1][0] == 1):
            return "LEFT"
        elif (curr_node[2][1]-curr_node[1][1] == -1):
            return "DOWN"
        elif (curr_node[2][1]-curr_node[1][1] == 1):
            return "UP"

    def main_loop(self):
        state = self.game.get_state()
        print(state)
        self.game.direction = self.find_path(state)
        # print(self.game.direction)
        self.game.startGame()

if __name__ == "__main__":
    agent = Agent()

    # Test python stuff
    # history = [(0, (1, 1), (2, 2), 1), (10, (4, 2), (3, 1), 2), (5, (10, 7), (4, 5), 10)]
    # print((1, 2) in list(n[1] for n in history))

    while True:
        agent.main_loop()
        