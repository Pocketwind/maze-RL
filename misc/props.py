from enum import Enum

class Props(Enum):
    WALL='￦'
    PLAYER='＠'
    ITEM='＊'
    BLANK='　'
    TRAFFIC='！'
    ITEM_1='１'
    ITEM_2='２'
    ITEM_3='３'
    def __str__(self):
        return self.name