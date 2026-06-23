#%% Importing Config Tools
from .Module1 import Module1
from .Module2 import Module2
#%% Importing Config Packets
class Phase0:
    class Modules:
        Module1 = Module1
        Module2 = Module2
    class General:
        Activation = True
        Version = 0