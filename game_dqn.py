import gymnasium as gym
import numpy as np
import random,time,os,copy
from gymnasium.spaces import MultiDiscrete,Discrete
from enum import IntEnum,Enum

TRAFFIC_REWARD=-10
ITEM_REWARD=100
MOVE_REWARD=-1

class Actions(IntEnum):
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3
    def __str__(self):
        return self.name
class Props(Enum):
    WALL='￦'
    PLAYER='＠'
    ITEM='＊'
    BLANK='　'
    TRAFFIC='！'
    def __str__(self):
        return self.name
    
map=[[Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL],
    [Props.WALL,Props.PLAYER,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.ITEM,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.ITEM,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.ITEM,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.ITEM,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.ITEM,Props.BLANK,Props.BLANK,Props.BLANK,Props.BLANK,Props.ITEM,Props.WALL,Props.BLANK,Props.BLANK,Props.BLANK,Props.WALL],
    [Props.WALL,Props.BLANK,Props.BLANK,Props.WALL,Props.WALL,Props.ITEM,Props.BLANK,Props.WALL,Props.WALL,Props.WALL,Props.BLANK,Props.BLANK,Props.WALL],
    [Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL,Props.WALL]]

class Game_DQN(gym.Env):
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
        if self.check_item():
            reward=ITEM_REWARD
        else:
            reward=MOVE_REWARD
        if self.check_traffic():
            reward=TRAFFIC_REWARD
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
            display[i[0]][i[1]]=Props.ITEM
        for i in range(self.height):
            for k in range(self.width):
                print(display[i][k].value,end='')
            print()
    def map_initialization(self):
        count=0
        for i in range(self.height):
            for k in range(self.width):
                if self.map[i][k]==Props.PLAYER:
                    self.x=self.initial_x=i
                    self.y=self.initial_y=k
                    self.map[i][k]=Props.BLANK
                    count+=1
                elif self.map[i][k]==Props.ITEM:
                    self.item_position.append([i,k])
                    self.item+=1
                    self.map[i][k]=Props.BLANK
                    count+=1
                elif self.map[i][k]==Props.TRAFFIC:
                    self.traffics.append([i,k])
                    self.map[i][k]=Props.TRAFFIC
                else:
                    pass
        self.initial_item=self.item
        self.initial_item_position=copy.deepcopy(self.item_position)
        self.initial_traffics=copy.deepcopy(self.traffics)
        self.num_states=self.item*self.action_space.n*self.height*self.width
        self.observation_space=gym.spaces.Discrete(self.num_states)
        return count
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
        for i in self.item_position:
            if self.x==i[0] and self.y==i[1]:
                self.item_position.remove(i)
                self.item-=1
                return True
        else:
            return False
    def check_traffic(self):
        for i in self.traffics:
            if self.x==i[0] and self.y==i[1]:
                return True
            else:
                return False