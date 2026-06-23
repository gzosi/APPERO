#%% Importing Libreries
from termcolor import colored
#%% Importing Code Tools
from Drivers.Phase1.Module3 import Task1
from Drivers.Phase1.Module3 import Task2
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase1.Modules.Module3.General.Activation is True:
        print('... Module3:', colored( 'Running ℹ️ ', 'cyan'))
        Task1.main(Config)
        Task2.main(Config)
        print('... Module3:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase1.Modules.Module3.General.Activation is False:
        print('... Module3:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Module3 Switch (on/off) ❌')
    return