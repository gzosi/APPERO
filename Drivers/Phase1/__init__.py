#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase1 import Module1
from Drivers.Phase1 import Module2
from Drivers.Phase1 import Module3
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase1.General.Activation is True:
        print('.. Phase1:', colored( 'Running ℹ️ ', 'cyan'))
        Module1.main(Config)
        Module2.main(Config)
        Module3.main(Config)
        print('.. Phase1:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase1.General.Activation is False:
        print('.. Phase1:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Phase1 Switch (on/off) ❌')
    return