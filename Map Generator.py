import misc.map_generator as MapGenerator
import numpy as np
import os

map=[]
for i in range(10):
    map.append(MapGenerator.generate_random_map(width=15,height=15,num_items=5,num_traffics=10,num_walls=100))

for i in range(len(map)):
    file=open(f"map{i}.txt","w")
    for j in range(len(map[i])):
        for k in range(len(map[i][j])):
            file.write(str(map[i][j][k].value))
        file.write("\n")
    file.close()