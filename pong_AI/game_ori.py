# Not really worth it to make the AI lmao
import pygame
import sys
import numpy as np

from pygame.rect import Rect

GRID_SIZE = 10
BOARD_SIZE = (GRID_SIZE, GRID_SIZE*8)
BALL_SIZE = (GRID_SIZE, GRID_SIZE)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

BALL_SPEED = 3
PLAYER_SPEED = 3
pygame.init()

class Game:
    def __init__(self):
        self.width = 500
        self.height = 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        
        self.score = 0
        self.bot_score = 0
        
        self.move_down = False
        self.move_up = False

        self.reset()

    def reset(self):
        # Player
        self.player_x = GRID_SIZE*3
        self.player_y = self.height / 2 - (BOARD_SIZE[1] / 2)
        
        # Ball
        self.ball_x = self.width / 2
        self.ball_y = self.height / 2

        self.ball_velocity_x = -BALL_SPEED + 1
        self.ball_velocity_y = 0

        # CPU
        self.bot_x = self.width - GRID_SIZE*3
        self.bot_y = self.height / 2 - (BOARD_SIZE[1] / 2)

        self.start_game()
    
    def check_move(self):
        if (self.move_up and self.player_y > 0):
            self.player_y -= PLAYER_SPEED
        elif (self.move_down and self.player_y < self.height-BOARD_SIZE[1]):
            self.player_y += PLAYER_SPEED

    def move_ball(self):
        self.ball_x += self.ball_velocity_x
        self.ball_y += self.ball_velocity_y

        # hits player (use elif because only one is possible)
        if ((self.ball_x > self.player_x + GRID_SIZE - 4 and self.ball_x < self.player_x + GRID_SIZE + 4) and (self.ball_y > self.player_y and self.ball_y < self.player_y + BOARD_SIZE[1])):
            y_hit = self.ball_y - self.player_y
            # sin: 0-90-180 -> 0-1-0 -> 45-90-45
            degree = (1 + np.sin(y_hit/BOARD_SIZE[1] * np.pi)) * np.pi/4
            # draw the triangle thingy 
            self.ball_velocity_x = np.sin(degree) * BALL_SPEED
            self.ball_velocity_y = np.cos(degree) * BALL_SPEED

            if (y_hit > BOARD_SIZE[1] / 2): self.ball_velocity_y = -self.ball_velocity_y
            
        elif ((self.ball_x < self.bot_x + 4 and self.ball_x > self.bot_x - 4) and (self.ball_y > self.bot_y and self.ball_y < self.bot_y + BOARD_SIZE[1])):
            y_hit = self.ball_y - self.player_y
            degree = (1 + np.sin(y_hit/BOARD_SIZE[1] * np.pi)) * np.pi/4
            self.ball_velocity_x = -np.sin(degree) * BALL_SPEED
            self.ball_velocity_y = -np.cos(degree) * BALL_SPEED

            if (y_hit > BOARD_SIZE[1] / 2): self.ball_velocity_y = -self.ball_velocity_y

        # hits horizontal boundary
        elif (self.ball_x < 0):
            self.bot_score += 1
            if (not self.check_win()):
                self.reset()
            else:
                self.game_over()
        
        elif (self.ball_x > self.width-GRID_SIZE):
            self.score += 1
            if (not self.check_win()):
                self.reset()
            else:
                self.game_over()

        # hits vertical boundary
        elif (self.ball_y < 0 or self.ball_y > self.height):
            self.ball_velocity_y = -self.ball_velocity_y
        
        self.bot_y = self.ball_y - BOARD_SIZE[1] / 2

    def check_win(self):
        return self.score >= 10 or self.bot_score >= 10

    def game_over(self):
        print(f"BOT WIN, score: {self.score}, bot score: {self.bot_score}")
        pygame.quit()
        sys.exit()
    
    def update_UI(self):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, Rect(self.width/2, 0, 2, self.height)) # center line
        
        pygame.draw.rect(self.screen, WHITE, Rect(self.player_x, self.player_y, BOARD_SIZE[0], BOARD_SIZE[1]))
        pygame.draw.rect(self.screen, WHITE, Rect(self.bot_x, self.bot_y, BOARD_SIZE[0], BOARD_SIZE[1]))

        pygame.draw.circle(self.screen, WHITE, (self.ball_x, self.ball_y), GRID_SIZE/2)
        pygame.display.flip()

    def start_game(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.move_down = True
                    elif event.key == pygame.K_UP:
                        self.move_up = True
                
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        self.move_down = False
                    elif event.key == pygame.K_UP:
                        self.move_up = False
            
            self.check_move()
            self.move_ball()
            self.update_UI()

            pygame.time.Clock().tick(180)

if __name__ == "__main__":
    Game()