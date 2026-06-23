#%% Importing Libreries
import pandas as pd
import pickle
from pathlib import Path
from termcolor import colored
#%% Defining Subroutines
def engine(value, sizeVal, typeVal):
    ''' Campiona in modo indipendente i punti validi per ogni telecamera e restituisce un dizionario. '''
    ptsSet = {}
    for camera in value.columns:
        valid = value[camera].dropna()
        if typeVal is not None:
            filtered = valid[valid.apply(lambda val: val[1] in typeVal)]
        else:
            filtered = valid
        pts = filtered.apply(lambda val: val[0])
        if sizeVal is not None:
            n_samples = min(len(pts), sizeVal)
            ptsSet[camera] = pts.sample(n=n_samples).tolist()
        else:
            ptsSet[camera] = pts.tolist()
    return ptsSet
def structureExplorer(data, sizes, types):
    ''' Esplora ricorsivamente la struttura dei dati e applica l'engine ai DataFrame. '''
    newData = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            newData[key] = engine(value, sizes.get(key, None), types.get(key, None))
        elif isinstance(value, dict):
            newData[key] = structureExplorer(value, sizes.get(key, {}), types.get(key, {}))
        else:
            newData[key] = value       
    return newData
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task2.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task2.MetaData.OutputExt)
        dstRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.__name__ /
                   task_conf.__name__ /
                   task_conf.MetaData.OutputExt)
        if not srcRoot.exists():
            raise FileNotFoundError(f"Dati non trovati!")
        dstRoot.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = pd.read_pickle(srcRoot)
            sizes = task_conf.Settings.sizes
            types = task_conf.Settings.types
            newData = structureExplorer(data, sizes, types)
            with open(dstRoot, "wb") as f:
                pickle.dump(newData, f)
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return