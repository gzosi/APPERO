#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase4 import Module1
from Drivers.Phase4 import Module2
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase4.General.Activation is True:
        print('.. Phase4:', colored( 'Running ℹ️ ', 'cyan'))
        Module1.main(Config)
        Module2.main(Config)
        print('.. Phase4:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase4.General.Activation is False:
        print('.. Phase4:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Phase4 Switch (on/off) ❌')
    return