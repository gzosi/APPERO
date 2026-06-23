#%% Importing Libreries
import inspect
import runpy
import shutil 
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
def destroyerCheck(config_file):
    ''' Verifica se il file di configurazione contiene l'attributo Destroyer a True. '''
    try:
        mod = runpy.run_path(str(config_file))
    except Exception:
        return None
    classes = [v for v in mod.values() if inspect.isclass(v)]
    if len(classes) != 1:
        return None
    MainClass = classes[0]
    try:
        return MainClass.General.Destroyer
    except AttributeError:
        return None
def destroyerRoot(relative_path_str, root_str):
    ''' Elimina la struttura di directory e pulisce le cartelle padre vuote. '''
    root = Path(root_str)
    relative_path = Path(relative_path_str).with_suffix("")
    full_path = root / relative_path
    if full_path.exists() and full_path.is_dir():
        shutil.rmtree(full_path) 
    current = full_path.parent
    archive_root = root.resolve()
    while current != archive_root and current.exists():
        try:
            current.rmdir() 
            current = current.parent
        except OSError:
            break
    return
#%% Defining Main Function
def main(Config):
    if Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task2.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
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
            destroyer = destroyerCheck(config_file)
            if destroyer is True:
                relative_target = Path(Config.Paths.CodeRoots.DriversRoot) / driver_rel_path
                destroyerRoot(relative_target, archive_root)
        print('.... Task2:', colored('Executed ✅', 'green'))
    elif Config.Packages.Drivers.Phases.Phase0.Modules.Module1.Tasks.Task2.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return