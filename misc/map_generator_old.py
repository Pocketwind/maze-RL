from misc import Actions,Props
import numpy as np
import random,copy
class MapGenerator:
    def generate_random_map(width=18,height=18,num_items=5,num_traffics=10,num_walls=100):
        while True:
            map = np.full((height, width), Props.BLANK)  # 18x18 맵을 모두 빈 공간으로 초기화

            # 벽을 주변에 배치 (상하좌우 벽으로 둘러쌓임)
            for i in range(height):
                map[i][0] = Props.WALL
                map[i][width-1] = Props.WALL
            for i in range(width):
                map[0][i] = Props.WALL
                map[height-1][i] = Props.WALL

            # 벽을 무작위로 배치
            for _ in range(num_walls):
                x, y = random.randint(1, height-1), random.randint(1, width-1)  # 주변 벽 내부에 배치
                map[x, y] = Props.WALL

            # 아이템을 무작위로 배치
            for _ in range(num_items):
                x, y = random.randint(1, height-1), random.randint(1, width-1)  # 주변 벽 내부에 배치
                map[x, y] = Props.ITEM

            for _ in range(num_traffics):
                x, y = random.randint(1, height-2), random.randint(1, width-2)
                map[x, y] = Props.TRAFFIC

            # 플레이어 시작 위치
            player_x, player_y = random.randint(1, height-1), random.randint(1, width-1)  # 주변 벽 내부에 배치
            map[player_x, player_y] = Props.PLAYER

            # 미로 생성 조건 검사
            # PLAYER 위치에서 모든 ITEM에 도달 가능한지 확인
            if is_maze_solvable(map, (player_x, player_y),num_items):
                return map

    def is_maze_solvable(map, start,num_items):
        width=len(map[0])
        height=len(map)
        # 미로 탐색에 사용할 재귀 함수
        def dfs(x, y):
            if x < 0 or x >= height-1 or y < 0 or y >= width-1 or map[x][y] == Props.WALL or visited[x][y]:
                return
            visited[x][y] = True
            if map[x][y] == Props.ITEM:
                items_collected.add((x, y))
            dfs(x + 1, y)
            dfs(x - 1, y)
            dfs(x, y + 1)
            dfs(x, y - 1)

        visited = [[False] * (width-1) for _ in range(height-1)]
        items_collected = set()
        dfs(start[0], start[1])
        return len(items_collected) == num_items