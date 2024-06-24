import numpy as np
import random
from game_v1 import *
import os,time

alpha = 0.1
gamma = 0.95
epsilon = 0.8
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000

map=[[Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL],
    [Props.WALL,Props.PLAYER,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.ITEM,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL],
    [Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL]]


game = Game_v1(map=map)
state_size = game.num_states
action_size = game.action_space.n
q_values = np.zeros([game.height,game.width,action_size])


def select_action(state):
    if np.random.rand() <= epsilon:
        action = game.action_space.sample(game.action_mask())
    else:
        q=copy.deepcopy(q_values[state[0]][state[1]])
        for i in range(len(q)):
            if game.action_mask()[i]==0:
                q[i]=-np.inf
        action=np.argmax(q)
    return action


for episode in range(50):
    state = game.reset()
    done = False
    total_reward = 0
    while not done:
        action = select_action(state)\
        
        next_state, reward, done, _ = game.step(action)
        old_value = q_values[state[0]][state[1]][action]
        next_max = np.max(q_values[next_state[0]][next_state[1]])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_values[state[0]][state[1]][action] = new_value
        
        state = next_state
        total_reward += reward


    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


    print("Episode {}: Total Reward = {}, Epsilon = {:.2f}".format(episode, total_reward, epsilon))

for actions in range(action_size):
    print(Actions(actions))
    for i in range(game.height):
        for j in range(game.width):
            print(round(q_values[i][j][actions],1),end="\t")
        print()
print(q_values.shape)

input("Press enter to test...")

total_epochs, total_penalties = 0, 0
episodes = 100
frames = []

while True:
    state = game.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        q=copy.deepcopy(q_values[state[0]][state[1]])
        for i in range(len(q)):
            if game.action_mask()[i]==0:
                q[i]=-np.inf
        action=np.argmax(q)
        state, reward, done, info = game.step(action)

        if reward == -10:
            penalties += 1
        
        os.system('cls')
        game.render()
        time.sleep(0.1)
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs