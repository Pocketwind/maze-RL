import numpy as np
import random
from game_v2 import *
import os,time
import matplotlib.pyplot as plt
from misc.props import *
from misc.actions import *
import copy
import misc.map_generator as MapGenerator


"""file=open("map/map0.txt","r")
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
file.close()"""

W=Props.WALL
I=Props.ITEM
B=Props.BLANK
T=Props.TRAFFIC
P=Props.PLAYER

map=[[W,W,W,W,W,W,W,W,W,W,W,W,W,W],
     [W,W,B,I,W,W,W,W,W,W,W,W,W,W],
     [W,W,B,W,W,B,B,B,I,W,W,W,W,W],
     [W,W,B,W,B,B,W,W,B,W,W,W,W,W],
     [W,W,B,W,B,W,W,W,B,W,W,W,W,W],
     [W,W,B,W,B,W,B,B,B,B,B,W,I,W],
     [W,B,B,B,B,B,B,W,W,W,B,B,B,W],
     [W,B,W,W,W,W,P,W,W,W,W,W,W,W],
     [W,I,W,W,W,W,W,W,W,W,W,W,W,W],
     [W,W,W,W,W,W,W,W,W,W,W,W,W,W]]


alpha = 0.1
gamma = 0.9
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000


game = Game_v2(map=map)
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

input("Press enter to test...")
while True:
    score=0
    state = game.reset()
    done = False

    while not done:
        os.system('cls')
        game.render()
        print(f"Score : {score}, Items : {game.initial_item-game.item}/{game.initial_item}")
        time.sleep(0.2)

        q=copy.deepcopy(q_values[state[0]][state[1]][state[2]])
        for i in range(len(q)):
            if game.action_mask()[i]==0:
                q[i]=-np.inf
        action=np.argmax(q)
        state, reward, done, info = game.step(action)
        score+=reward