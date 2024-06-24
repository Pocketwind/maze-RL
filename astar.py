from game_v2 import *
import os,time,random
import sys,pygame
from pygame.locals import *

start=time.time()

TARGET_FPS = 2
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BOARD_WIDTH=15
BOARD_HEIGHT=15

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

game = Game_v2(map=map)

class Node:
    def __init__(self,parent,position):
        self.parent=parent
        self.position=position
        self.f=0
        self.g=0
        self.h=0
    def __eq__(self,other):
        return self.position==other.position
def astar(maze,start,end):
    start_node=Node(None,start)
    end_node=Node(None,end)
    open_list=[]
    closed_list=[]
    open_list.append(start_node)
    while open_list:
        current_node=open_list[0]
        current_index=0
        for index,item in enumerate(open_list):
            if item.f<current_node.f:
                current_node=item
                current_index=index
        open_list.pop(current_index)
        closed_list.append(current_node)
        if current_node==end_node:
            path=[]
            current=current_node
            while current is not None:
                path.append(current.position)
                current=current.parent
            return path[::-1]
        children=[]
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position=(
                current_node.position[0]+new_position[0],
                current_node.position[1]+new_position[1]
            )
            within_range=[
                node_position[0]>len(maze)-1,
                node_position[0]<0,
                node_position[1]>len(maze[len(maze)-1])-1,
                node_position[1]<0
            ]
            if any(within_range):
                continue
            if maze[node_position[0]][node_position[1]]!=Props.BLANK:
                continue
            new_node=Node(current_node,node_position)
            children.append(new_node)
        for child in children:
            if child in closed_list:
                continue
            child.g=current_node.g+1
            child.h=((child.position[0]-end_node.position[0])**2)+((child.position[1]-end_node.position[1])**2)
            child.f=child.g+child.h
            if len([open_node for open_node in open_list if child==open_node and child.g>open_node.g])>0:
                continue
            open_list.append(child)

def prop_to_screen(list,screen,brick):
    for i in range(BOARD_WIDTH):
        for j in range(BOARD_HEIGHT):
            if list[i][j]==Props.WALL:
                screen.blit(brick,(i*WINDOW_WIDTH/BOARD_WIDTH,j*WINDOW_HEIGHT/BOARD_HEIGHT))
            elif list[i][j]==Props.PLAYER:
                pygame.draw.ellipse(screen,(255,255,255),(i*WINDOW_WIDTH/BOARD_WIDTH,j*WINDOW_HEIGHT/BOARD_HEIGHT,WINDOW_WIDTH/BOARD_WIDTH,WINDOW_HEIGHT/BOARD_HEIGHT))
            elif list[i][j]==Props.TRAFFIC:
                pygame.draw.ellipse(screen,(255,0,0),(i*WINDOW_WIDTH/BOARD_WIDTH,j*WINDOW_HEIGHT/BOARD_HEIGHT,WINDOW_WIDTH/BOARD_WIDTH,WINDOW_HEIGHT/BOARD_HEIGHT))
            elif list[i][j]==Props.ITEM:
                pygame.draw.ellipse(screen,(0,255,255),(i*WINDOW_WIDTH/BOARD_WIDTH,j*WINDOW_HEIGHT/BOARD_HEIGHT,WINDOW_WIDTH/BOARD_WIDTH,WINDOW_HEIGHT/BOARD_HEIGHT))  


def main():
    destinations=[]
    destinations.append((game.x,game.y))
    for i in range(len(game.item_position)):
        destinations.append((game.item_position[i][0],game.item_position[i][1]))

    print(destinations)

    shortest=[]
    shortest.append(destinations[0])
    destinations.pop(0)
    start=0
    while len(destinations)!=0:
        distances=[]
        for x,y in destinations:
            distances.append((x-shortest[-1][0])**2+(y-shortest[-1][1])**2)
        shortest.append(destinations[distances.index(min(distances))])
        destinations.pop(distances.index(min(distances)))

    brick=pygame.image.load("misc/sprite/brick.png")
    brick=pygame.transform.scale(brick,(WINDOW_WIDTH/BOARD_WIDTH,WINDOW_HEIGHT/BOARD_HEIGHT))
    clock=pygame.time.Clock()

    pygame.init()
    screen=pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
    pygame.display.set_caption("GridWorld with Q-Learning")

    print(shortest)

    while True:
        for i in range(len(shortest)-1):
            start=shortest[i]
            end=shortest[i+1]
            path=astar(map,start,end)
            for x,y in path:
                temp=copy.deepcopy(map)
                temp_shortest=copy.deepcopy(shortest)
                for k in range(i+1):
                    temp_shortest.pop(0)
                temp[x][y]=Props.PLAYER
                for k in temp_shortest:
                    temp[k[0]][k[1]]=Props.ITEM
                screen.fill((0,0,0))
                for event in pygame.event.get():
                    if event.type==QUIT:
                        pygame.quit()
                        sys.exit()
                prop_to_screen(temp,screen,brick)
                clock.tick(TARGET_FPS)
                pygame.display.update()

main()

print(time.time()-start)