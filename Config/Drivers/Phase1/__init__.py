#%% Importing Config Tools
from .Module1 import Module1
from .Module2 import Module2
from .Module3 import Module3
#%% Importing Config Packets
class Phase1:
    class Modules:
        Module1 = Module1
        Module2 = Module2
        Module3 = Module3
    class General:
        Activation = False
        Version = 0