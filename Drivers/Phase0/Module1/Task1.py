#%% Importing Libreries
import inspect
import runpy
from pathlib import Path
from termcolor import colored
#%% Defining Subroutines
def driverExplorer(drivers_root):
    ''' Esplora l'albero delle directory di drivers_root e trova tutti i file Python (.py). '''
    driver_files = []
    root = Path(drivers_root)
    for file_path in root.rglob('*.py'):
        rel_path = file_path.relative_to(root)
        driver_files.append((str(file_path), str(rel_path)))
    return driver_files
def makerCheck(config_file):
    ''' Verifica se il file di configurazione contiene una classe con l'attributo Maker a True. '''
    try:
        mod = runpy.run_path(str(config_file))
    except Exception:
        return None
    classes = [v for v in mod.values() if inspect.isclass(v)]
    if len(classes) != 1:
        return None
    MainClass = classes[0]
    try:
        return MainClass.General.Maker
    except AttributeError:
        return None
def rootMaker(relative_path_str, root_str):
    ''' Crea la struttura di directory in base al percorso relativo. '''
    root = Path(root_str)
    relative_path = Path(relative_path_str)
    if relative_path.suffix:
        relative_path = relative_path.with_suffix("")
    full_path = root / relative_path
    full_path.mkdir(parents=True, exist_ok=True)
    return
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task1.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        drivers_root = main_root / Config.Paths.CodeRoots.DriversRoot
        config_root = main_root / Config.Paths.CodeRoots.ConfigRoot
        archive_root = (main_root / 
                        Config.Paths.DataRoots.ResourcesRoot / 
                        Config.Paths.DataRoots.StreamRoot / 
                        Config.Paths.DataRoots.CaseStudyRoot())
        driver_files = driverExplorer(drivers_root)
        for _, driver_rel_path in driver_files:
            config_file = config_root / Config.Paths.CodeRoots.DriversRoot / driver_rel_path
            maker = makerCheck(config_file)
            if maker is True:
                relative_target = Path(Config.Paths.CodeRoots.DriversRoot) / driver_rel_path
                rootMaker(relative_target, archive_root)
        print('.... Task1:', colored('Executed ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task1.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return