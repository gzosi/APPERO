#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase0.Module1 import Task1
from Drivers.Phase0.Module1 import Task2
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase0.Modules.Module1.General.Activation is True:
        print('... Module1:', colored( 'Running ℹ️ ', 'cyan'))
        Task1.main(Config)
        Task2.main(Config)
        print('... Module1:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase0.Modules.Module1.General.Activation is False:
        print('... Module1:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Module1 Switch (on/off) ❌')
    return