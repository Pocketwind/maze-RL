import numpy as np
import random
from game_v3 import *
import os,time
import matplotlib.pyplot as plt
from misc.props import *
from misc.actions import *
import copy
import misc.map_generator as MapGenerator
import pygame
from pygame.locals import *


B=Props.BLANK
I1=Props.ITEM_1
I2=Props.ITEM_2
I3=Props.ITEM_3
T=Props.TRAFFIC
W=Props.WALL
P=Props.PLAYER

B=Props.BLANK
I=Props.ITEM
I1=Props.ITEM_1
I2=Props.ITEM_2
I3=Props.ITEM_3
T=Props.TRAFFIC
W=Props.WALL
P=Props.PLAYER

map=[[W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W],
     [W,B,B,B,B,W,B,B,B,B,B,B,I1,W,W,B,W,W,W,W],
     [W,B,W,W,W,B,B,W,W,W,W,W,W,W,W,B,W,W,W,W],
     [W,B,W,W,W,B,B,I2,B,B,W,W,W,W,W,B,W,W,W,W],
     [W,B,W,W,W,B,W,W,W,T,W,W,B,B,B,B,W,W,W,W],
     [W,I1,B,B,B,B,B,B,B,B,B,B,B,W,W,B,B,W,W,W],
     [W,B,W,W,W,B,W,W,W,B,W,W,W,W,W,W,B,B,I3,W],
     [W,B,W,W,W,B,W,W,B,B,W,W,W,W,W,W,W,W,W,W],
     [W,B,W,W,W,B,W,W,B,W,W,W,W,W,W,W,W,W,W,W],
     [W,B,B,B,B,B,B,B,B,B,B,B,B,B,B,P,W,W,W,W],
     [W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W,W]]
alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000


game = Game_v3(map=map)
game.render()

input("Press enter to start...")


start=time.time()

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

data={"steps":[],"epsilon":[],"reward":[]}

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
    
    data["steps"].append(steps)
    data["epsilon"].append(epsilon)
    data["reward"].append(total_reward)

    print("Episode {}: Total Reward = {}, Epsilon = {:.2f}, Steps = {}".format(episode, total_reward, epsilon, steps))

print("Training time : {} seconds".format(time.time()-start))

x=np.arange(0,num_episodes)
plt.subplot(2,2,1)
plt.plot(x,data["steps"],label="Steps",color="dodgerblue")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.subplot(2,2,2)
plt.plot(x,data["epsilon"],label="Epsilon",color="green")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()
plt.subplot(2,2,3)
plt.plot(x,data["reward"],label="Reward",color="orange")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()

def prop_to_screen(list):
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if list[i][j]==Props.WALL:
                screen.blit(brick,(j*KAN,i*KAN))
            elif list[i][j]==Props.ITEM_1:
                pygame.draw.ellipse(screen,(0,60,60),pygame.Rect(j*KAN,i*KAN,KAN,KAN))
            elif list[i][j]==Props.ITEM_2:
                pygame.draw.ellipse(screen,(0,160,160),pygame.Rect(j*KAN,i*KAN,KAN,KAN))
            elif list[i][j]==Props.ITEM_3:
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