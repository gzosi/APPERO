#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase3 import Module1
from Drivers.Phase3 import Module2
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase3.General.Activation is True:
        print('.. Phase3:', colored( 'Running ℹ️ ', 'cyan'))
        Module1.main(Config)
        Module2.main(Config)
        print('.. Phase3:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase3.General.Activation is False:
        print('.. Phase3:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Phase3 Switch (on/off) ❌')
    return