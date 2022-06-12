# Imports
from numpy.lib.function_base import place
import pygame
import sys
import numpy as np

# Constants
GRID_SIZE = 20
BODY_SIZE = 18
SCREEN_SIZE = (500, 500)
SCREEN_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)

pygame.init()

font = pygame.font.SysFont('Roboto-Medium.ttf', 25)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((500, 500))
        self.snake = [(80, 20), (60, 20), (40, 20), (20, 20)]

        self.x = 4
        self.y = 1
        self.score = 0
        self.direction = 'RIGHT'

        self.placeFood()

    def drawSnake(self):
        for body in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, pygame.Rect(body[0], body[1], BODY_SIZE, BODY_SIZE))
    
    def moveSnake(self):
        if (self.direction == 'RIGHT'):
            self.x += 1
        elif (self.direction == 'LEFT'):
            self.x -= 1
        elif (self.direction == 'UP'):
            self.y -= 1
        elif (self.direction == 'DOWN'):
            self.y += 1

        self.snake.insert(0, (self.x * GRID_SIZE, self.y * GRID_SIZE))
        pygame.draw.rect(self.screen, SCREEN_COLOR, pygame.Rect(self.snake[0][0], self.snake[0][1], BODY_SIZE, BODY_SIZE))
        self.snake.pop()

        if (self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]):
            self.placeFood()
            self.snake.append(self.snake[-1])
            self.score = self.score + 1

    def placeFood(self):
        self.food = (np.random.randint(0, 24) * GRID_SIZE, np.random.randint(0, 24) * GRID_SIZE)
        while (self.food in self.snake):
            self.food = (np.random.randint(0, 24) * GRID_SIZE, np.random.randint(0, 24) * GRID_SIZE)

    def checkFinished(self):
        if (self.x < 0 or self.y < 0 or self.x > 24 or self.y > 24 or (self.x * GRID_SIZE, self.y * GRID_SIZE) in self.snake[1:]):
            pygame.quit()
            print("FINISHED, SCORE: ", self.score)
            sys.exit()
        
    def startGame(self):
        # event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(SCREEN_COLOR)
        # draw food
        pygame.draw.rect(self.screen, FOOD_COLOR, pygame.Rect(self.food[0], self.food[1], BODY_SIZE, BODY_SIZE))    
        
        self.moveSnake()
        self.drawSnake()
        self.checkFinished()
        
        pygame.time.Clock().tick(15)
        text = font.render("Score: " + str(self.score), True, '#ffffff')
        self.screen.blit(text, [0, 0])
        pygame.display.flip()
    
    def get_state(self):
        return (self.direction, [tuple(coord // GRID_SIZE for coord in body) for body in self.snake], tuple(coord // GRID_SIZE for coord in self.food))

if __name__ == '__main__':           
    Game()


#d