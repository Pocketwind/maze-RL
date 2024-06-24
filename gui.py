import sys,pygame
from pygame.locals import *
from game_v2 import *
from misc.props import *
from misc.actions import *
import copy
import os,time
import misc.map_generator as MapGenerator
import numpy as np
import random




alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000

file=open("map/map9.txt","r")
map=[[0 for j in range(15)] for i in range(15)]
row=0
for line in file:
    for i in range(len(line)):
        if line[i]=="\n":
            continue
        if line[i]==Props.BLANK.value:
            map[row][i]=Props.BLANK
        elif line[i]==Props.ITEM.value:
            map[row][i]=Props.ITEM
        elif line[i]==Props.TRAFFIC.value:
            map[row][i]=Props.TRAFFIC
        elif line[i]==Props.WALL.value:
            map[row][i]=Props.WALL
        elif line[i]==Props.PLAYER.value:
            map[row][i]=Props.PLAYER
    row+=1
file.close()

B=Props.BLANK
I=Props.ITEM
I1=Props.ITEM_1
I2=Props.ITEM_2
I3=Props.ITEM_3
T=Props.TRAFFIC
W=Props.WALL
P=Props.PLAYER

map=[[W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W],
     [W,B,B,B,B,W,B,B,B,B,B,B,I,W,W,B,W,W,W,W],
     [W,B,W,W,W,B,B,W,W,B,W,W,W,W,W,B,W,W,W,W],
     [W,B,W,W,W,B,B,I,B,B,W,W,W,W,W,B,W,W,W,W],
     [W,B,W,W,W,B,W,W,W,T,W,W,B,B,B,B,W,W,W,W],
     [W,B,B,B,B,B,B,B,B,B,B,B,B,W,W,B,B,W,W,W],
     [W,I,W,W,W,B,W,W,W,B,W,W,W,W,W,W,B,B,I,W],
     [W,B,W,W,W,B,W,W,B,B,W,W,W,W,W,W,W,W,W,W],
     [W,B,W,W,W,B,W,W,B,W,W,W,W,W,W,W,W,W,W,W],
     [W,B,B,B,B,B,B,B,B,B,B,B,B,B,B,P,W,W,W,W],
     [W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W]]


game = Game_v2(map=map)
state_size = game.num_states
action_size = game.action_space.n 
items=game.item
q_values = np.zeros([game.height,game.width,items,action_size])


def select_action(state):
    if np.random.rand() <= epsilon:
        action = game.action_space.sample(game.action_mask())
    else:
        q=copy.deepcopy(q_values[state[0]][state[1]][state[2]])
        for i in range(len(q)):
            if game.action_mask()[i]==0:
                q[i]=-np.inf
        action=np.argmax(q)
    return action

for episode in range(num_episodes):
    state = game.reset()
    done = False
    total_reward = 0
    steps=0
    while not done:
        action = select_action(state)
        
        next_state, reward, done, info = game.step(action)
        old_value = q_values[state[0]][state[1]][state[2]][action]
        next_max = np.max(q_values[next_state[0]][next_state[1]][next_state[2]])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_values[state[0]][state[1]][state[2]][action] = new_value
        
        state = next_state
        total_reward += reward

        steps+=1

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print("Episode {}: Total Reward = {}, Epsilon = {:.2f}, Steps = {}".format(episode, total_reward, epsilon, steps))



def prop_to_screen(list):
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if list[i][j]==Props.WALL:
                screen.blit(brick,(j*KAN,i*KAN))
            elif list[i][j]==Props.ITEM:
                pygame.draw.ellipse(screen,(0,255,255),pygame.Rect(j*KAN,i*KAN,KAN,KAN))
            elif list[i][j]==Props.TRAFFIC:
                pygame.draw.ellipse(screen,(255,0,0),pygame.Rect(j*KAN,i*KAN,KAN,KAN))
            elif list[i][j]==Props.PLAYER:
                pygame.draw.ellipse(screen,(255,255,255),pygame.Rect(j*KAN,i*KAN,KAN,KAN))
KAN=50
TARGET_FPS = 2
WINDOW_WIDTH = KAN*len(map[0])
WINDOW_HEIGHT = KAN*len(map)
BOARD_WIDTH=len(map[0])
BOARD_HEIGHT=len(map)

brick=pygame.image.load("misc/sprite/brick.png")
brick=pygame.transform.scale(brick,(WINDOW_WIDTH/BOARD_WIDTH,WINDOW_HEIGHT/BOARD_HEIGHT))
clock=pygame.time.Clock()

pygame.init()
screen=pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
pygame.display.set_caption("GridWorld with Q-Learning")

state=game.reset()
running=True
while running:
    screen.fill((0,0,0))
    for event in pygame.event.get():
        if event.type==QUIT:
            running=False
    
    if done:
        state=game.reset()
    q=copy.deepcopy(q_values[state[0]][state[1]][state[2]])
    for i in range(len(q)):
        if game.action_mask()[i]==0:
            q[i]=-np.inf
    action=np.argmax(q)
    state, reward, done, info = game.step(action)

    prop_to_screen(game.get_map())
    clock.tick(TARGET_FPS)
    pygame.display.update()
pygame.quit()