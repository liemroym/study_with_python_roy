from random import randint
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F

pygame.init()

SCREEN_POS = (0, 0)
SCREEN_SIZE = (400, 400) 
SCREEN_COLOR = (0, 0, 0)
GRID_AMOUNT = (20, 20) # (0->(X-1), 0->(Y-1))
NODE_SIZE = (SCREEN_SIZE[0] // GRID_AMOUNT[0], SCREEN_SIZE[1] // GRID_AMOUNT[1]) # (0, SS - NS)
INTERVAL = 100

class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()

        self.node_count = 100
        self.nodes : list[Node] = []

        self.counter = 0

        self.init_node()
        self.main()

    def init_node(self):
        for i in range(self.node_count):
            self.create_node()

    def create_node(self, model=None):
        pos = (randint(0, GRID_AMOUNT[0]-1) * NODE_SIZE[0], randint(0, GRID_AMOUNT[1]-1) * NODE_SIZE[1])

        while (self.screen.get_at((pos[0], pos[1])) != SCREEN_COLOR):
            pos = (randint(0, GRID_AMOUNT[0]-1) * NODE_SIZE[0], randint(0, GRID_AMOUNT[1]-1) * NODE_SIZE[1])

        self.nodes.append(Node(self.screen, pos, model))
        self.nodes[len(self.nodes)-1].draw()
        

    def update_UI(self):
        self.screen.fill(SCREEN_COLOR)
        for node in self.nodes:
            node.draw()
        
        pygame.display.flip()

    def delete_half(self):
        i = 0
        count = 0
        while i < len(self.nodes):
            node = self.nodes[i]
            if (node.x < SCREEN_SIZE[0] // 2):
                self.nodes.pop(i)
                i -= 1
                count += 1
            i += 1

        temp = self.nodes.copy()
        self.nodes.clear()
        for i in range(len(temp)):
            if len(self.nodes) < (GRID_AMOUNT[0] * GRID_AMOUNT[1]) - 4:
                self.create_node(temp[i].model)
                self.create_node(temp[i].model)

        print(count)


    def main(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.update_UI()
            
            for node in self.nodes:
                node.do_move()

            self.counter += 1
            if (self.counter > INTERVAL):
                self.counter = 0
                self.delete_half()
                self.node_count = len(self.nodes)
                print(self.node_count)

            

class Node:
    def __init__(self, surface, pos, model=None):
        self.surface : pygame.Surface = surface
        self.x = pos[0]
        self.y = pos[1]

        if (model == None):
            self.model = LinearNet(4, 4, 5)
        else:
            self.model = model

    def do_move(self):
        percepts = [int(self.coll_top()), int(self.coll_bottom()), int(self.coll_left()), int(self.coll_right())]
        input = torch.tensor(percepts, dtype=torch.long)

        res = self.model(input.float())

        move = torch.argmax(res)
    
        if (move == 0):
            self.move_top()
        elif (move == 1):
            self.move_bottom()
        elif (move == 2):
            self.move_left()
        elif (move == 3):
            self.move_right()
            



    # Check collision
    def coll_top(self):
        searched_node = (self.x, self.y - NODE_SIZE[1])
        if (searched_node[1] >= SCREEN_POS[1]):
            if (self.surface.get_at(searched_node) == SCREEN_COLOR):
                return False
        
        return True
    
    def coll_bottom(self):
        searched_node = (self.x, self.y + NODE_SIZE[1])
        if (searched_node[1] < SCREEN_SIZE[1]):
            if (self.surface.get_at(searched_node) == SCREEN_COLOR):
                return False
        
        return True

    def coll_left(self):
        searched_node = (self.x - NODE_SIZE[0], self.y)
        if (searched_node[0] >= SCREEN_POS[0]):
            if (self.surface.get_at(searched_node) == SCREEN_COLOR):
                return False
        
        return True

    def coll_right(self):
        searched_node = (self.x + NODE_SIZE[0], self.y)
        if (searched_node[0] < SCREEN_SIZE[0]):
            if (self.surface.get_at(searched_node) == SCREEN_COLOR):
                return False
        
        return True
    
    def move_top(self):
        if (not self.coll_top()):
            self.y -= NODE_SIZE[1]

    def move_bottom(self):
        if (not self.coll_bottom()):
            self.y += NODE_SIZE[1]

    def move_left(self):
        if (not self.coll_left()):
            self.x -= NODE_SIZE[0]

    def move_right(self):
        if (not self.coll_right()):
            self.x += NODE_SIZE[0]
    
    def draw(self):
        pygame.draw.rect(self.surface, (255, 255, 255), pygame.Rect(self.x, self.y, NODE_SIZE[0], NODE_SIZE[1]))



class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # print("1", x)
        # print(x, x.type())
        # print(input, input.type())
        x = self.linear1(input)
        # print("2", x)
        x = self.linear2(x)
        # print("3", x)

        return x

if __name__ == "__main__":
    Simulation()



