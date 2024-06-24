import gymnasium as gym
import numpy as np
import random,time,os,copy
from misc.props import *
from misc.actions import *



ITEM_REWARD_1=1000
ITEM_REWARD_2=ITEM_REWARD_1*2
ITEM_REWARD_3=ITEM_REWARD_1*3
TRAFFIC_REWARD=-int(ITEM_REWARD_1*0.2)
MOVE_REWARD=-1

class Game_v3(gym.Env):
    def __init__(self,**kwargs):
        super().__init__()
        self.map=kwargs.get('map',map)
        self.item=0
        self.x=0
        self.initial_x=0
        self.y=0
        self.initial_y=0
        self.item_position=[]
        self.initial_item_position=[]
        self.initial_item=0
        self.traffics=[]
        self.initial_traffics=[]
        self.width=len(self.map[0])
        self.height=len(self.map)
        self.num_states=0
        self.action_space=gym.spaces.Discrete(4)
        self.map_initialization()
    def get_state(self):
        return np.array([self.x,self.y,self.item-1])
    def take_action(self,action):
        if action==Actions.UP:
            self.x=self.x-1
        elif action==Actions.DOWN:
            self.x=self.x+1
        elif action==Actions.LEFT:
            self.y=self.y-1
        elif action==Actions.RIGHT:
            self.y=self.y+1
    def step(self,action):
        self.take_action(action)
        reward=-1
        reward=self.check_item()
        if self.check_traffic():
            reward+=TRAFFIC_REWARD
        done=self.is_done()
        state=self.get_state()
        return state,reward,done,None
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.x=self.initial_x
        self.y=self.initial_y
        self.item=self.initial_item
        self.item_position=copy.deepcopy(self.initial_item_position)
        self.traffics=copy.deepcopy(self.initial_traffics)
        return self.get_state()
    def render(self):
        display=copy.deepcopy(self.map)
        display[self.x][self.y]=Props.PLAYER
        for i in self.item_position:
            if i[1]==ITEM_REWARD_1:
                display[i[0][0]][i[0][1]]=Props.ITEM_1
            elif i[1]==ITEM_REWARD_2:
                display[i[0][0]][i[0][1]]=Props.ITEM_2
            elif i[1]==ITEM_REWARD_3:
                display[i[0][0]][i[0][1]]=Props.ITEM_3
        for i in range(self.height):
            for k in range(self.width):
                print(display[i][k].value,end='')
            print()
    def map_initialization(self):
        for i in range(self.height):
            for k in range(self.width):
                if self.map[i][k]==Props.PLAYER:
                    self.x=self.initial_x=i
                    self.y=self.initial_y=k
                    self.map[i][k]=Props.BLANK
                elif self.map[i][k]==Props.ITEM_1:
                    self.item_position.append(([i,k],ITEM_REWARD_1))
                    self.item+=1
                    self.map[i][k]=Props.BLANK
                elif self.map[i][k]==Props.ITEM_2:
                    self.item_position.append(([i,k],ITEM_REWARD_2))
                    self.item+=1
                    self.map[i][k]=Props.BLANK
                elif self.map[i][k]==Props.ITEM_3:
                    self.item_position.append(([i,k],ITEM_REWARD_3))
                    self.item+=1
                    self.map[i][k]=Props.BLANK
                elif self.map[i][k]==Props.TRAFFIC:
                    self.traffics.append([i,k])
                else:
                    pass
        self.initial_item=self.item
        self.initial_item_position=copy.deepcopy(self.item_position)
        self.initial_traffics=copy.deepcopy(self.traffics)
        self.num_states=self.item*2*self.action_space.n
        self.observation_space=gym.spaces.Discrete(self.num_states)
        return
    def action_mask(self):
        move_available=np.array([0,0,0,0],dtype=np.int8)
        #UP
        if self.map[self.x-1][self.y]==Props.WALL:
            move_available[0]=0
        else:
            move_available[0]=1
        #DOWN
        if self.map[self.x+1][self.y]==Props.WALL:
            move_available[1]=0
        else:
            move_available[1]=1
        #LEFT
        if self.map[self.x][self.y-1]==Props.WALL:
            move_available[2]=0
        else:   
            move_available[2]=1
        #RIGHT
        if self.map[self.x][self.y+1]==Props.WALL:
            move_available[3]=0
        else:
            move_available[3]=1
        return move_available
    def is_done(self):
        if self.item==0:
            return True
        else:
            return False
    def check_item(self):
        for i in range(len(self.item_position)):
            pos,reward=self.item_position[i]
            if self.x==pos[0] and self.y==pos[1]:
                self.item_position.remove(self.item_position[i])
                self.item-=1
                return reward
        else:
            return -1
    def check_traffic(self):
        ret=False
        for i in self.traffics:
            if self.x==i[0] and self.y==i[1]:
                ret=True
                break
            else:
                ret=False
        return ret
    def get_map(self):
        display=copy.deepcopy(self.map)
        display[self.x][self.y]=Props.PLAYER
        for i in range(len(self.item_position)):
            pos,reward=self.item_position[i]
            if reward==ITEM_REWARD_1:
                display[pos[0]][pos[1]]=Props.ITEM_1
            elif reward==ITEM_REWARD_2:
                display[pos[0]][pos[1]]=Props.ITEM_2
            elif reward==ITEM_REWARD_3:
                display[pos[0]][pos[1]]=Props.ITEM_3
        return display