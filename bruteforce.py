from game_v2 import *
import os,time,random

start=time.time()

B=Props.BLANK
I=Props.ITEM
T=Props.TRAFFIC
W=Props.WALL
P=Props.PLAYER

map=[[W,W,W,W,W,W,W,W,W,W,W,W],
     [W,P,W,B,B,B,W,B,B,B,B,W],
     [W,B,W,I,W,B,W,B,W,W,I,W],
     [W,B,W,W,B,B,B,B,B,W,W,W],
     [W,B,T,B,B,W,W,W,B,W,I,W],
     [W,B,W,B,W,B,B,B,B,B,B,W],
     [W,B,B,B,W,B,W,B,W,W,W,W],
     [W,B,W,W,W,B,W,B,W,B,I,W],
     [W,I,W,I,B,B,W,B,B,B,W,W],
     [W,W,W,W,W,W,W,W,W,W,W,W]]

"""map=[[W,W,W,W,W,W,W,W,W,W,W],
     [W,B,B,B,B,B,W,B,I,B,W],
     [W,B,B,W,B,B,W,B,B,B,W],
     [W,B,B,B,B,B,W,B,B,W,W],
     [W,B,B,W,W,B,B,B,B,B,W],
     [W,P,B,B,W,W,B,B,B,B,W],
     [W,W,W,W,W,W,W,W,W,W,W]]"""

game = Game_v2(map=map)

REPEATS=1000
def main():
    game.render()
    input("Press enter to start...")


    best_reward=-np.inf
    best_path=[]
    for i in range(REPEATS):
        state = game.reset()
        done = False
        total_reward=0
        best_path=[]
        while not done:
            action = game.action_space.sample(game.action_mask())
            best_path.append(action)
            _, reward, done, _ = game.step(action)
            total_reward += reward

        print(f"Episode {i+1} : {total_reward}, Steps : {len(best_path)}")
        if total_reward>best_reward:
            best_reward=total_reward
            best_path=copy.deepcopy(best_path)

    total_steps=len(best_path)
    
    print(f"Best Reward : {best_reward}, Steps : {total_steps}")

    input("Press enter to test...")
    state = game.reset()
    done = False
    while True:
        score=0
        while not done:
            os.system("cls")
            print(f"Best Reward : {best_reward}, Steps : {total_steps-len(best_path)}/{total_steps}")
            action = best_path.pop(0)
            _, reward, done, _ = game.step(action)
            game.render()
            score+=reward
            print(f"Score : {score}")
            time.sleep(0.0)
main()

print(time.time()-start)