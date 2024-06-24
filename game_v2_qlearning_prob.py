import numpy as np
import random
from game_v2 import *
import os,time
import matplotlib.pyplot as plt


file=open("map/map0.txt","r")
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

alpha = 0.1
gamma = 0.95
epsilon = 0.8
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000
    
game = Game_v2(map=map)
state_size = game.num_states
action_size = game.action_space.n 
items=game.item
q_values = np.zeros([game.height,game.width,items,action_size])

game.render()

input("Press enter to start...")

''' probability algorithm '''

def safe_softmax(q, tau):
    max_q = np.max(q)
    exp_q = np.exp((q - max_q) / tau)
    probabilities = exp_q / np.sum(exp_q)
    return probabilities

def select_action(state, tau):
    q = copy.deepcopy(q_values[state[0]][state[1]][state[2]])
    for i in range(len(q)):
        if game.action_mask()[i] == 0:
            q[i] = -np.inf
    probabilities = safe_softmax(q, tau)  # safe_softmax 함수로 확률 계산
    action = np.random.choice(len(probabilities), p=probabilities)
    return action

initial_tau = 1.0
final_tau = 0.01
tau_decay = 0.995

tau = initial_tau

data={"steps":[],"epsilon":[],"reward":[]}

start=time.time()

for episode in range(num_episodes):
    state = game.reset()
    done = False
    total_reward = 0
    steps=0
    while not done:
        action = select_action(state, tau)
        #os.system('cls')
        # game.render()
        next_state, reward, done, _ = game.step(action)
        
        old_value = q_values[state[0]][state[1]][state[2]][action]
        next_max = np.max(q_values[next_state[0]][next_state[1]][next_state[2]])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_values[state[0]][state[1]][state[2]][action] = new_value
        
        state = next_state
        total_reward += reward

        steps+=1

    if tau > final_tau:
        tau *= tau_decay

    data["steps"].append(steps)
    data["epsilon"].append(tau)
    data["reward"].append(total_reward)

    print("Episode {}: Total Reward = {}, Steps = {}".format(episode, total_reward, steps))


print("Training time : {} seconds".format(time.time()-start))
x=np.arange(0,num_episodes)
plt.subplot(2,2,1)
plt.plot(x,data["steps"],label="Steps",color="dodgerblue")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.legend()
plt.subplot(2,2,2)
plt.plot(x,data["epsilon"],label="Tau",color="green")
plt.xlabel("Episode")
plt.ylabel("Tau")
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

        action=select_action(state,tau=0.01)
        state, reward, done, info = game.step(action)
        score+=reward