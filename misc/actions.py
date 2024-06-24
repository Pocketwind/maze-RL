from enum import IntEnum

class Actions(IntEnum):
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3
    def __str__(self):
        return self.name
