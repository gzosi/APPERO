#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase2.Module2 import Task1
from Drivers.Phase2.Module2 import Task2
from Drivers.Phase2.Module2 import Task3
from Drivers.Phase2.Module2 import Task4
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase2.Modules.Module2.General.Activation is True:
        print('... Module2:', colored( 'Running ℹ️ ', 'cyan'))
        Task1.main(Config)
        Task2.main(Config)
        Task3.main(Config)
        Task4.main(Config)
        print('... Module2:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase2.Modules.Module2.General.Activation is False:
        print('... Module2:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Module2 Switch (on/off) ❌')
    return