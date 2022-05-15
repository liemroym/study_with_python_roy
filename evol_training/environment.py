from math import floor
from random import randint
import pygame
import numpy as np

import tensorflow as tf
# import tensorflow as tf

# tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 
# tf.test.is_gpu_available()
# tf.__version__
pygame.init()

SCREEN_POS = (0, 0)
SCREEN_SIZE = (400, 400) 
SCREEN_COLOR = (0, 0, 0)
GRID_AMOUNT = (20, 20) # (0->(X-1), 0->(Y-1))
NODE_SIZE = (SCREEN_SIZE[0] // GRID_AMOUNT[0], SCREEN_SIZE[1] // GRID_AMOUNT[1]) # (0, SS - NS)
INTERVAL = 50
RANDOM_VAR = 0
iterator = 0

class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()

        self.node_count = 50
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
            if (node.x > SCREEN_SIZE[0] // 2):
                self.nodes.pop(i)
                i -= 1
                count += 1
            i += 1

        temp = self.nodes.copy()
        self.nodes.clear()
        self.screen.fill(SCREEN_COLOR)
        print(len(temp))
        for i in range(len(temp)):
            j = randint(0, len(temp)-1)
            if len(self.nodes) < (GRID_AMOUNT[0] * GRID_AMOUNT[1] // 4):
                self.create_node(temp[j].model)
                self.create_node(temp[j].model)
                temp.pop(j)
            else:
                break

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
            self.model = LinearNet(4, 4)
        else:
            self.model = model
        
        self.model.build(input_shape=(1, 4))

        # if (self.model.linear2 != None):
        #     self.color = self.model.linear2.weight
        # else:
        #     self.color = self.model.linear1.weight

        # weight_list = self.color.tolist() 
        # self.color = (
        #     # R = w0
        #     floor(sum(weight_list[0]) / 5 * 128) + 128,
        #     # G = w1 + w3//2
        #     floor(sum(weight_list[1]) / 5 * 64) + (sum(weight_list[3]) / 5 * 64) + 128,
        #     # B = w2 + w3//2
        #     floor(sum(weight_list[2]) / 5 * 64) + (sum(weight_list[3]) / 5 * 64) + 128
        # )

    def do_move(self):
        move = [0, 0, 0, 0]
        if (randint(0, iterator) < RANDOM_VAR):
            move[randint(0, 4)] = 1
        else: 
            # percepts = [int(self.coll_top()), int(self.coll_bottom()), int(self.coll_left()), int(self.coll_right())]
            percepts = [[int(self.coll_top()), int(self.coll_bottom()), int(self.coll_left()), int(self.coll_right())]]
            input = tf.Variable(percepts)

            res = self.model(input)
            res.numpy()
            
            res = res[0]
            max = 0
            maxIdx = 0
            for i, act in enumerate(res):
                if act >= max:
                    max = act
                    maxIdx = i

            move[maxIdx] = 1

        if (move[0]):
            self.move_top()
        elif (move[1]):
            self.move_bottom()
        elif (move[2]):
            self.move_left()
        elif (move[3]):
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
        # pygame.draw.rect(self.surface, self.color, pygame.Rect(self.x, self.y, NODE_SIZE[0], NODE_SIZE[1]))
        pygame.draw.rect(self.surface, (255,255,255), pygame.Rect(self.x, self.y, NODE_SIZE[0], NODE_SIZE[1]))


class LinearNet(tf.keras.Model):
    def __init__(self, input_size, output_size, hidden_size=None):
        if (hidden_size == None):
            super(LinearNet, self).__init__()
            self.layer1 = tf.keras.layers.Dense(output_size, input_shape=(input_size,))
            self.activation = tf.keras.layers.Activation('sigmoid')
            
        else:
            # self.linear1 = nn.Linear(input_size, output_size)
            # self.linear2 = None
            pass

    def call(self, input):
        # print(input)
        input = self.layer1(input)
        input = self.activation(input)
        return input        

if __name__ == "__main__":
    Simulation()
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    # model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # tf.keras.layers.Dense(4096,activation='relu'),
    # tf.keras.layers.Dense(4096,activation='relu'),
    # tf.keras.layers.Dense(10, activation='softmax')
    # ])
    # model.summary()

    # model.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'],)

    # model.fit(np.expand_dims(x_train,3), y_train, epochs=2, batch_size=1024)

