#%% Importing Config Tools
from .Task1 import Task1
from .Task2 import Task2
from .Task3 import Task3
#%% Importing Config Packets
class Module2:
    class Tasks:
        Task1 = Task1
        Task2 = Task2
        Task3 = Task3
    class General:
        Activation = True
        Version = 0