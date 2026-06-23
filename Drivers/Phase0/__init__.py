#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase0 import Module1
from Drivers.Phase0 import Module2
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase0.General.Activation is True:
        print('.. Phase0:', colored( 'Running ℹ️ ', 'cyan'))
        Module1.main(Config)
        Module2.main(Config)
        print('.. Phase0:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase0.General.Activation is False:
        print('.. Phase0:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Phase0 Switch (on/off) ❌')
    return