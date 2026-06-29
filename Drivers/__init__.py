#%% Importing Libreries
from termcolor import colored
#%% Importing Code Phases
from Drivers import Phase0
from Drivers import Phase1
from Drivers import Phase2
from Drivers import Phase3
from Drivers import Phase4
#%% Defining Main Function
def main(Config):
    print('----------------------------------')
    print('----------------------------------')
    if Config.Packages.Drivers.General.Activation is True:
        print('. Drivers:', colored( 'Running ℹ️ ', 'cyan'))
        print('----------------------------------')
        Phase0.main(Config)
        print('----------------------------------')
        Phase1.main(Config)
        print('----------------------------------')
        Phase2.main(Config)
        print('----------------------------------')
        Phase3.main(Config)
        print('----------------------------------')
        Phase4.main(Config)
        print('----------------------------------')
        print('. Drivers:', colored( 'Exexuted ✅', 'green'))
    elif Config.Packages.Drivers.General.Activation is False:
        print('. Drivers:',colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Plesas Set the Drivers Switch (on/off) ❌')
    print('----------------------------------')
    print('----------------------------------')
    return
