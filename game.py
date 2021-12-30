# Imports
from numpy.lib.function_base import place
import pygame
import sys
import numpy as np

# Constants
GRID_SIZE = 20
BODY_SIZE = 18
SCREEN_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)

pygame.init()
font = pygame.font.Font('Roboto-Medium.ttf', 25)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((500, 500))
        # head = snake[0], tail = snake[-1]
        self.snake = [(80, 20), (60, 20), (40, 20), (20, 20)]
        self.reset()

    def reset(self):
        # starting condition
        self.x = 4
        self.y = 1
        self.score = 0
        self.direction = 'RIGHT'
        
        self.frameIteration = 0

        self.placeFood()

    # draw all of the snake body
    def drawSnake(self):
        for body in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, pygame.Rect(body[0], body[1], BODY_SIZE, BODY_SIZE))
    
    # move snake according to direction (action) given
    def moveSnake(self, action):
        # action = [straight, right, left] = [bool(int), bool(int), bool(int)]
        clockwise = ['RIGHT', 'DOWN', 'LEFT', 'UP'] # clockwise rotation, (+) = right, (-) = left
        idx = clockwise.index(self.direction) # get direction index to rotate

        if (action[0] == 1):
            newDirection = self.direction
        elif (action[1] == 1): # right
            newDirection = self.direction[(idx+1) % 4]
        else: # left, action[2] == 1
            newDirection = self.direction[(idx-1) % 4]

        # move according to newDirection
        if (newDirection == 'RIGHT'):
            self.x += 1
        elif (newDirection == 'LEFT'):
            self.x -= 1
        elif (newDirection == 'UP'):
            self.y -= 1
        elif (newDirection == 'DOWN'):
            self.y += 1

        self.snake.insert(0, (self.x * GRID_SIZE, self.y * GRID_SIZE))
        pygame.draw.rect(self.screen, SCREEN_COLOR, pygame.Rect(self.snake[0][0], self.snake[0][1], BODY_SIZE, BODY_SIZE))
        self.snake.pop()

    # places food on random location other than snake location
    def placeFood(self):
        self.food = (np.random.randint(0, 24) * GRID_SIZE, np.random.randint(0, 24) * GRID_SIZE)
        while (self.food in self.snake):
            self.food = (np.random.randint(0, 24) * GRID_SIZE, np.random.randint(0, 24) * GRID_SIZE)
        pygame.draw.rect(self.screen, FOOD_COLOR, pygame.Rect(self.food[0], self.food[1], BODY_SIZE, BODY_SIZE))    

    def checkCollision(self, point=None):
        # receives point, checks if the snake will collide if it went to that point
        if (point == None): point = self.snake[0]
        x = point[0] / 20
        y = point[1] / 20
        return (x < 0 or y < 0 or x > 24 or y > 24 or (x * GRID_SIZE, y * GRID_SIZE) in self.snake[1:])
        
    def startGame(self, action):
        reward = 0
        gameOver = False
        self.frameIteration += 1
        # event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.moveSnake(action)
        self.drawSnake()
        
        # finish if colliding / game went too long on that snake length
        if (self.checkCollision or self.frameIteration > 100 * len(self.snake)):
            gameOver = True
            reward = -10
            return reward, gameOver, self.score
        
        # snake eats food
        if (self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]):
            self.placeFood()
            self.snake.append(self.snake[-1])
            self.score += 1
            reward = 10

        # game will wait for 15 frames
        pygame.time.Clock().tick(15)
        pygame.display.flip()
        
        return reward, gameOver, self.score 