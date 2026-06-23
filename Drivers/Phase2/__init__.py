#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase2 import Module1
from Drivers.Phase2 import Module2
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase2.General.Activation is True:
        print('.. Phase2:', colored( 'Running ℹ️ ', 'cyan'))
        Module1.main(Config)
        Module2.main(Config)
        print('.. Phase2:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase2.General.Activation is False:
        print('.. Phase2:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Phase2 Switch (on/off) ❌')
    return