# For rotation system, refer to https://tetris.fandom.com/wiki/SRS
# All the rotation are pre-determined following the wiki above (not using matrix calculation)

# Conclusion: I suck at geometry. I suck at naming. I suck at coding in general

from asyncio.windows_events import NULL
import random
import pygame

pygame.init()

GRID_SIZE = 20
PIECE_SIZE = 18

SCREEN_SIZE = (800, 600)

HOLD_PIECE_POSITION = (40, 40) # Black part
PIECE_BOARD_SIZE = (6 * GRID_SIZE, 4 * GRID_SIZE)

GAME_BOARD_POSITION = ((HOLD_PIECE_POSITION[0] + PIECE_BOARD_SIZE[0] + GRID_SIZE) + GRID_SIZE, HOLD_PIECE_POSITION[1] - GRID_SIZE)
GAME_BOARD_SIZE = (10 * GRID_SIZE, 20 * GRID_SIZE)

NEXT_PIECE_POSITION = ((GAME_BOARD_POSITION[0] + GAME_BOARD_SIZE[0] + GRID_SIZE) + GRID_SIZE, GAME_BOARD_POSITION[1] + GRID_SIZE)


NEXT_PIECE = 5

SCREEN_COLOR = (0, 0, 0)
GAME_BOARD_COLOR = (50, 50, 50)

PIECES = ['t', 'z', 's', 'j', 'l', 'i', 'o']

PIECE_COLOR = {
    't': (134, 1, 175),  # purple
    'z': (255, 0, 0),    # red
    's': (0, 255, 0),    # green
    'j': (0, 0, 255),    # blue
    'l': (255, 165, 0),  # orange
    'i': (64, 244, 208), # aqua
    'o': (255, 255, 0),  # yellow
}

GHOST_COLOR = {
    't': (104, 0, 155), # purple
    'z': (175, 0, 0), # red
    's': (0, 175, 0), # green
    'j': (0, 0, 175), # blue
    'l': (205, 115, 0), # orange
    'i': (14, 194, 158), # aqua
    'o': (175, 175, 0) # yellow
}

# Each index represents the rotation state (refer to tetris wiki on SRS), index 0 is the spawn state
# Pieces were read from left to right, for each column do top to bottom reading. 
# Rotation doesn't use matrix calculation

PIECE_SHAPE = {
    't': [[(0, 1), (1, 0), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 1)], [(0, 1), (1, 1), (1, 2), (2, 1)], [(0, 1), (1, 0), (1, 1), (1, 2)]], 
    'z': [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 1), (1, 2), (2, 0), (2, 1)], [(0, 1), (1, 1), (1, 2), (2, 2)], [(0, 1), (0, 2), (1, 0), (1, 1)]], 
    's': [[(0, 1), (1, 0), (1, 1), (2, 0)], [(1, 0), (1, 1), (2, 1), (2, 2)], [(0, 2), (1, 1), (1, 2), (2, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]],
    'j': [[(0, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 0)], [(0, 1), (1, 1), (2, 1), (2, 2)], [(0, 2), (1, 0), (1, 1), (1, 2)]],
    'l': [[(0, 1), (1, 1), (2, 0), (2, 1)], [(1, 0), (1, 1), (1, 2), (2, 2)], [(0, 1), (0, 2), (1, 1), (2, 1)], [(0, 0), (1, 0), (1, 1), (1, 2)]],
    'i': [[(0, 1), (1, 1), (2, 1), (3, 1)], [(2, 0), (2, 1), (2, 2), (2, 3)], [(0, 2), (1, 2), (2, 2), (3, 2)], [(1, 0), (1, 1), (1, 2), (1, 3)]],
    'o': [[(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 0), (1, 1), (2, 0), (2, 1)]],
}

# For centering the piece
PIECE_WIDTH = {
    't': 3,
    'z': 3,
    's': 3,
    'j': 3,
    'l': 3,
    'i': 4,
    'o': 2
}

# SRS wall kicks. Each row represents these state changes:
# 0>>1
# 1>>0
# 1>>2
# 2>>1
# 2>>3
# 3>>2
# 3>>0
# 0>>3

OTHER_WALL_KICK = [[(-1, 0), (-1, 1), ( 0,-2), (-1,-2)],
                   [( 1, 0), ( 1,-1), ( 0, 2), ( 1, 2)],
                   [( 1, 0), ( 1,-1), ( 0, 2), ( 1, 2)],
                   [(-1, 0), (-1, 1), ( 0,-2), (-1,-2)],
                   [( 1, 0), ( 1, 1), ( 0,-2), ( 1,-2)],
                   [(-1, 0), (-1,-1), ( 0, 2), (-1, 2)],
                   [(-1, 0), (-1,-1), ( 0, 2), (-1, 2)],
                   [( 1, 0), ( 1, 1), ( 0,-2), ( 1,-2)]]

L_WALL_KICK = [[(-2, 0), ( 1, 0), (-2,-1), ( 1, 2)],
               [( 2, 0), (-1, 0), ( 2, 1), (-1,-2)],
               [(-1, 0), ( 2, 0), (-1, 2), ( 2,-1)],
               [( 1, 0), (-2, 0), ( 1,-2), (-2, 1)],
               [( 2, 0), (-1, 0), ( 2, 1), (-1,-2)],
               [(-2, 0), ( 1, 0), (-2,-1), ( 1, 2)],
               [( 1, 0), (-2, 0), ( 1,-2), (-2, 1)],
               [(-1, 0), ( 2, 0), (-1, 2), ( 2,-1)]]

FALL_COUNTER = 20
LOCK_COUNTER = 30
DAS_COUNTER = 10

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        
        self.current_bag = PIECES.copy()
        random.shuffle(self.current_bag)
        self.next_bag = PIECES.copy()
        random.shuffle(self.next_bag)
        
        self.current_tetromino = Tetromino(self.screen, self.current_bag.pop())
        self.hold_tetromino = NULL
        self.held = False

        self.soft_drop_controller = False
        self.DAS_left_controller = False
        self.DAS_right_controller = False

        self.tetrominoes = [self.current_tetromino]
        
        self.gravity_counter = 0
        self.lock_counter = 0
        self.DAS_counter = 0

        self.update_ghost()
        self.main()

    def main(self):
        # forever loop
        running = True
        while running:
            # event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_x:
                        self.current_tetromino.rotate_cw()
                    elif event.key == pygame.K_z:
                        self.current_tetromino.rotate_ccw()
                    elif event.key == pygame.K_DOWN:
                        self.soft_drop_controller = True
                    elif event.key == pygame.K_UP:
                        self.current_tetromino.hard_drop()
                        self.lock_tetromino()
                    elif event.key == pygame.K_LEFT:
                        self.DAS_left_controller = True
                        self.current_tetromino.move_left()
                    elif event.key == pygame.K_RIGHT:
                        self.DAS_right_controller = True
                        self.current_tetromino.move_right()
                    elif event.key == pygame.K_c:
                        if (not self.held):
                            last_hold_tetromino = self.hold_tetromino.type if self.hold_tetromino != NULL else NULL
                            if (self.next_bag == []):
                                self.next_bag = PIECES.copy()
                                random.shuffle(self.next_bag)
                            self.held = True
                            self.hold_tetromino = Tetromino(self.screen, self.current_tetromino.type, x=HOLD_PIECE_POSITION[0] + 2 * GRID_SIZE, y=HOLD_PIECE_POSITION[1] + 2 * GRID_SIZE)    
                            self.tetrominoes.remove(self.current_tetromino)
                            self.current_tetromino = Tetromino(self.screen, self.current_bag.pop(0) if last_hold_tetromino == NULL else last_hold_tetromino) 
                            self.current_bag.append(self.next_bag.pop())
                            self.tetrominoes.append(self.current_tetromino)
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN:
                        self.soft_drop_controller = False
                    elif event.key == pygame.K_LEFT:
                        self.DAS_counter = 0
                        self.DAS_left_controller = False
                    elif event.key == pygame.K_RIGHT:
                        self.DAS_counter = 0
                        self.DAS_right_controller = False

            self.check_move()
            self.handle_fall()
            self.update_UI()
            self.update_ghost()
            self.current_tetromino.draw()

            pygame.display.flip()
            pygame.time.Clock().tick(60)

    def update_UI(self):
        self.screen.fill(SCREEN_COLOR)
        # Hold piece
        pygame.draw.rect(self.screen, GAME_BOARD_COLOR, pygame.Rect(HOLD_PIECE_POSITION[0] - GRID_SIZE, HOLD_PIECE_POSITION[1] - GRID_SIZE, PIECE_BOARD_SIZE[0] + GRID_SIZE * 2, PIECE_BOARD_SIZE[1] + GRID_SIZE * 2))
        pygame.draw.rect(self.screen, SCREEN_COLOR, pygame.Rect(HOLD_PIECE_POSITION[0], HOLD_PIECE_POSITION[1], PIECE_BOARD_SIZE[0], PIECE_BOARD_SIZE[1]))
        if (self.hold_tetromino != NULL):
            self.hold_tetromino.draw()

        # Game board
        pygame.draw.rect(self.screen, GAME_BOARD_COLOR, pygame.Rect(GAME_BOARD_POSITION[0], GAME_BOARD_POSITION[1], GAME_BOARD_SIZE[0], GAME_BOARD_SIZE[1]))

        # Next piece
        pygame.draw.rect(self.screen, GAME_BOARD_COLOR, pygame.Rect(NEXT_PIECE_POSITION[0] - GRID_SIZE, NEXT_PIECE_POSITION[1] - GRID_SIZE, PIECE_BOARD_SIZE[0] + GRID_SIZE * 2, PIECE_BOARD_SIZE[1] * NEXT_PIECE - ((NEXT_PIECE-1) * GRID_SIZE) + GRID_SIZE * 2))
        pygame.draw.rect(self.screen, SCREEN_COLOR, pygame.Rect(NEXT_PIECE_POSITION[0], NEXT_PIECE_POSITION[1], PIECE_BOARD_SIZE[0], PIECE_BOARD_SIZE[1] * NEXT_PIECE - ((NEXT_PIECE-1) * GRID_SIZE)))
        for i in range(NEXT_PIECE):
            Tetromino(self.screen, self.current_bag[i], NEXT_PIECE_POSITION[0] + GRID_SIZE, (NEXT_PIECE_POSITION[1] * (i+1) + 2 * GRID_SIZE) + (GRID_SIZE * i)).draw()

        # Tetrominoes
        for tetromino in self.tetrominoes:
            tetromino.draw()

    def check_move(self):
        if (self.soft_drop_controller and not self.current_tetromino.check_collision_bottom()):
            self.current_tetromino.fall()
        
        if (self.DAS_left_controller):
            self.DAS_counter += 1
            if (self.DAS_counter >= DAS_COUNTER):
                self.current_tetromino.move_left()
        
        if (self.DAS_right_controller):
            self.DAS_counter += 1
            if (self.DAS_counter >= DAS_COUNTER):
                self.current_tetromino.move_right()

    def handle_fall(self):
        self.gravity_counter += 1
        
        if (self.gravity_counter == FALL_COUNTER):
            self.current_tetromino.fall()
            self.gravity_counter = 0

        if (self.current_tetromino.bottom):
            self.lock_counter += 1
        else:
            self.lock_counter = 0

        if (self.lock_counter == LOCK_COUNTER):
            self.lock_tetromino()

    def lock_tetromino(self):
        self.lock_counter = 0
        self.held = False
        self.check_line_clear()
        if (self.next_bag == []):
            self.next_bag = PIECES.copy() 
            random.shuffle(self.next_bag)

        self.current_tetromino = Tetromino(self.screen, self.current_bag.pop(0))
        self.current_bag.append(self.next_bag.pop())
        self.tetrominoes.append(self.current_tetromino)

    def check_line_clear(self):
        minoes_in_line : list[list[tuple(Tetromino, Mino)]]= []

        for i in range(23):
            minoes_in_line.append([])

        for tetromino in self.tetrominoes:
            for mino in tetromino.minoes:
                minoes_in_line[mino.y // 20].append((tetromino, mino))
        
        line_full = []

        for j, line in enumerate(minoes_in_line):
            if (len(line) == 10):
                line_full.append(j)
            elif (j == 0 and len(line) != 0):
                print("DEAD")
        
        for line in line_full:
            for cleared_mino_pair in minoes_in_line[line]:
                cleared_mino_pair[0].minoes.remove(cleared_mino_pair[1])
            
            for line_falling in minoes_in_line[:line]:
                for mino_pair in line_falling:
                    mino_pair[1].y += GRID_SIZE
    
    def update_ghost(self):
        self.ghost_tetromino = Tetromino(self.screen, self.current_tetromino.type, x=self.current_tetromino.x, y=self.current_tetromino.y, ghost_state=self.current_tetromino.state)
        self.ghost_tetromino.hard_drop()
        self.ghost_tetromino.draw()

# Tetrominoes: pieces, minoes: tiles, official names prob
class Mino:
    def __init__(self, x, y, screen, color):
        self.x = x
        self.y = y
        self.screen = screen
        self.color = color
        self.width = PIECE_SIZE
        self.height = PIECE_SIZE
        
    def draw(self):
        pygame.draw.rect(self.screen, self.color, pygame.Rect(self.x, self.y, self.width, self.height))

    def check_collision_inside(self, coords):
        if ((self.x, self.y) not in coords and self.y > GAME_BOARD_POSITION[1]):
            inside_color = self.screen.get_at((self.x + GRID_SIZE // 2, self.y + GRID_SIZE // 2)) 
            if (inside_color != GAME_BOARD_COLOR and inside_color not in GHOST_COLOR.values()):
                return True
        return False

    def check_collision_bottom(self, coords):
        # Check if bottom part of the piece collide with something or not
        # Don't check colission with fellow pieces
        if ((self.x, self.y + GRID_SIZE) not in coords):
            bottom_color = self.screen.get_at((self.x + (GRID_SIZE // 2), self.y + GRID_SIZE + (GRID_SIZE // 2))) 
            if (self.y + GRID_SIZE > GAME_BOARD_SIZE[1]):
                return True
            elif (bottom_color != GAME_BOARD_COLOR and bottom_color not in GHOST_COLOR.values()): 
                return True
        return False
        
    def check_collision_left(self, coords):
        # Check if left part of the piece collide with something or not
        # Don't check colission with fellow pieces
        if ((self.x - GRID_SIZE, self.y) not in coords):
            left_color = self.screen.get_at((self.x - GRID_SIZE + (GRID_SIZE // 2), self.y + (GRID_SIZE // 2))) 
            if (self.x - GRID_SIZE < GAME_BOARD_POSITION[0]):
                return True
            elif (self.y >= GAME_BOARD_POSITION[1]):
                if (left_color != GAME_BOARD_COLOR and left_color not in GHOST_COLOR.values()): 
                    return True
        return False
        
    def check_collision_right(self, coords):
        # Check if right part of the piece collide with something or not
        # Don't check colission with fellow pieces
        if ((self.x + GRID_SIZE, self.y) not in coords):
            right_color = self.screen.get_at((self.x + GRID_SIZE + (GRID_SIZE // 2), self.y + (GRID_SIZE // 2))) 
            if (self.x + GRID_SIZE > GAME_BOARD_POSITION[0] + GAME_BOARD_SIZE[0] - GRID_SIZE):
                return True
            elif (self.y >= GAME_BOARD_POSITION[1]):
                if (right_color != GAME_BOARD_COLOR and right_color not in GHOST_COLOR.values()): 
                    return True
        return False
        
class Tetromino:
    def __init__(self, screen, type, x=None, y=None, ghost_state=None):
        self.x = ((GAME_BOARD_POSITION[0] + GAME_BOARD_SIZE[0] // 2) - 2 * GRID_SIZE) if x == None else x
        self.y = GAME_BOARD_POSITION[1] if y == None else y
        self.screen = screen
        self.bottom = False
        self.type = type    
        self.minoes : list[Mino] = []
        self.coords = []

        if (ghost_state == None):
            self.state = 0 # for rotating. 0 for spawn state, 0>1>2>3 (refer to the SRS wall kick naming convention)    
            self.color = PIECE_COLOR[type]
        else:
            self.state = ghost_state # for rotating. 0 for spawn state, 0>1>2>3 (refer to the SRS wall kick naming convention)    
            self.color = GHOST_COLOR[type]
        self.update()
        
    def draw(self):
        for mino in self.minoes:
            mino.draw()

    def update(self):
        self.minoes.clear()
        self.coords.clear()
        for shape in PIECE_SHAPE[self.type][self.state]:
            self.minoes.append(Mino((shape[0]) * GRID_SIZE + self.x, (shape[1] - 1) * GRID_SIZE + self.y, self.screen, self.color))
            self.coords.append((shape[0] * GRID_SIZE + self.x, (shape[1] - 1) * GRID_SIZE + self.y))
            
    def check_collision_bottom(self):
        for mino in self.minoes:
            if (mino.check_collision_bottom(self.coords)):
                self.bottom = True
                return True

        # If SRS wall kick - kicks in, the piece would still fall 
        self.bottom = False
        return False

    def fall(self):
        if (not self.check_collision_bottom()):  
            self.y += GRID_SIZE
            self.update()

    def check_collision_left(self):
        for mino in self.minoes:
            if (mino.check_collision_left(self.coords)):
                return True
        return False

    def move_left(self):
        if (not self.check_collision_left()):
            self.x -= GRID_SIZE
            self.update()

    def check_collision_right(self):
        for mino in self.minoes:
            if (mino.check_collision_right(self.coords)):
                return True
        return False

    def move_right(self):
        if (not self.check_collision_right()):
            self.x += GRID_SIZE
            self.update()

    def rotate_cw(self):
        prev_state = self.state
        prev_x = self.x
        prev_y = self.y
        curr_coords = self.coords.copy()
        self.state = (self.state + 1) % 4
        self.update()
        
        safe = False
        test = 0
        if (self.type != 'o'):
            while (not safe):
                if (test == 4): 
                    self.x = prev_x
                    self.y = prev_y
                    self.state -= 1
                    if (self.state == -1): self.state = 3
                    self.update()
                    break
                for mino in self.minoes:
                    if (mino.check_collision_inside(curr_coords)):
                        if (self.type == 'l'):
                            safe = False
                            self.x += L_WALL_KICK[prev_state * 2][test][0] * GRID_SIZE
                            self.y += L_WALL_KICK[prev_state * 2][test][1] * GRID_SIZE
                            test += 1
                            self.update()
                        else:
                            safe = False
                            self.x += OTHER_WALL_KICK[prev_state * 2][test][0] * GRID_SIZE
                            self.y += OTHER_WALL_KICK[prev_state * 2][test][1] * GRID_SIZE
                            test += 1
                            self.update()
                        break
                    safe = True

        pygame.display.flip()

    def rotate_ccw(self):
        prev_state = self.state
        prev_x = self.x
        prev_y = self.y
        curr_coords = self.coords.copy()
        self.state -= 1
        if (self.state == -1): self.state = 3
        self.update()
        
        safe = False
        test = 0
        if (self.type != 'o'):
            while (not safe):
                if (test == 4): 
                    self.x = prev_x
                    self.y = prev_y
                    self.state = (self.state + 1) % 4
                    self.update()
                    break
                for mino in self.minoes:
                    if (mino.check_collision_inside(curr_coords)):
                        if (self.type == 'l'):
                            safe = False
                            self.x += L_WALL_KICK[prev_state * 2][test][0] * GRID_SIZE
                            self.y += L_WALL_KICK[prev_state * 2][test][1] * GRID_SIZE
                            test += 1
                            self.update()
                        else:
                            safe = False
                            self.x += OTHER_WALL_KICK[prev_state * 2][test][0] * GRID_SIZE
                            self.y += OTHER_WALL_KICK[prev_state * 2][test][1] * GRID_SIZE
                            test += 1
                            self.update()
                        break
                    safe = True

        pygame.display.flip()

    def hard_drop(self):
        while (not self.bottom):
            self.fall()
        


if __name__ == "__main__":
    Game()
    