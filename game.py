# Imports
from numpy.lib.function_base import place
import pygame
import sys
import numpy as np
import time
from torch.autograd.grad_mode import F

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
        self.reset()

    def reset(self):
        # starting condition
        # head = snake[0], tail = snake[-1]
        self.snake = [(200, 200), (200, 180), (200, 160)]        
        
        self.x = 200
        self.y = 200
        self.score = 0
        self.direction = 'UP'

        self.frameIteration = 0

        self.placeFood()    
    
    # move snake according to direction (action) given
    def moveSnake(self, action):
        # action = [straight, right, left] = [bool(int), bool(int), bool(int)]
        clockwise = ['RIGHT', 'DOWN', 'LEFT', 'UP'] # clockwise rotation, (+) = right, (-) = left
        idx = clockwise.index(self.direction) # get direction index to rotate

        if (np.array_equal(action, [1, 0, 0])):
            newDirection = self.direction
        elif (np.array_equal(action, [0, 1, 0])): # right
            newIdx = (idx+1) % 4
            newDirection = clockwise[newIdx]
        else: # left, [0, 0, 1]
            newIdx = (idx-1) % 4
            newDirection = clockwise[newIdx]

        self.direction = newDirection
        # move according to newDirection
        if (newDirection == 'RIGHT'):
            self.x += 20
        elif (newDirection == 'LEFT'):
            self.x -= 20
        elif (newDirection == 'UP'):
            self.y -= 20
        elif (newDirection == 'DOWN'):
            self.y += 20

    # places food on random location other than snake location
    def placeFood(self):
        self.food = (np.random.randint(0, 24) * GRID_SIZE, np.random.randint(0, 24) * GRID_SIZE)
        while (self.food in self.snake):
            self.food = (np.random.randint(0, 24) * GRID_SIZE, np.random.randint(0, 24) * GRID_SIZE)

    def checkCollision(self, point=None):
        # receives point, checks if the snake will collide if it went to that point
        if (point == None): point = self.snake[0]
        x = point[0]
        y = point[1]
        return (x < 0 or y < 0 or x > 480 or y > 480 or (x, y) in self.snake[1:])
        
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
        self.snake.insert(0, (self.x, self.y))
        self.snake.pop()
        
        self.updateUI()
        pygame.display.flip()
        # finish if colliding / game went too long on that snake length
        if (self.checkCollision() or self.frameIteration > 100 * len(self.snake)):
            gameOver = True
            reward = -10
            pygame.draw.rect(self.screen, FOOD_COLOR, pygame.rect.Rect(self.snake[0][0], self.snake[0][1], BODY_SIZE, BODY_SIZE))
            pygame.display.flip()
            return reward, gameOver, self.score
        
        # snake eats food
        if (self.snake[0][0] == self.food[0] and self.snake[0][1] == self.food[1]):
            self.placeFood()
            self.snake.append(self.snake[-1])
            self.score += 1
            reward = 10

        # game will wait for 15 frames
        pygame.time.Clock().tick(40)
        
        return reward, gameOver, self.score 

    def updateUI(self):
        self.screen.fill(SCREEN_COLOR)

        for body in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, pygame.Rect(body[0], body[1], BODY_SIZE, BODY_SIZE))

        pygame.draw.rect(self.screen, FOOD_COLOR, pygame.Rect(self.food[0], self.food[1], BODY_SIZE, BODY_SIZE))    

        text = font.render("Score: " + str(self.score), True, '#ffffff')
        self.screen.blit(text, [0, 0])