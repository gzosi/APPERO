#%% Importing Config Tools
from .Task1 import Task1
from .Task2 import Task2
# from .Task4 import Task4
#%% Importing Config Packets
class Module1:
    class Tasks:
        Task1 = Task1
        Task2 = Task2
    class General:
        Activation = True
        Version = 0