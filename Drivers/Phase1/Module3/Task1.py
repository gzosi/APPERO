#%% Importing Libreries
import pandas as pd
import itertools
import pickle
from pathlib import Path
from termcolor import colored
#%% Defining Subroutines
def engine(value, sizeVal, typeVal):
    ''' Campiona punti validi per coppie di camere garantendo la sincronizzazione temporale. '''
    pairs = list(itertools.permutations(value.columns, 2))
    ptsSet = {}
    for pair in pairs:
        subset = value[list(pair)].copy()
        def is_valid_detection(x):
            if x is None: 
                return False
            if isinstance(x, float) and pd.isna(x):
                return False
            try:
                if len(x) != 2: 
                    return False
                pts, det_type = x
            except (TypeError, ValueError):
                return False 
            has_points = pts is not None and len(pts) > 0
            valid_type = (det_type in typeVal) if typeVal is not None else True
            return has_points and valid_type
        mask = subset.apply(lambda row: is_valid_detection(row[pair[0]]) and is_valid_detection(row[pair[1]]), axis=1)
        valid_subset = subset[mask]
        cam_a_pts = valid_subset[pair[0]].map(lambda x: x[0])
        cam_b_pts = valid_subset[pair[1]].map(lambda x: x[0])
        if sizeVal is not None and len(valid_subset) > sizeVal:
            sampled_indices = valid_subset.sample(n=sizeVal).index
            cam_a_pts = cam_a_pts.loc[sampled_indices]
            cam_b_pts = cam_b_pts.loc[sampled_indices]
        ptsSet[pair] = {
            pair[0]: cam_a_pts.tolist(),
            pair[1]: cam_b_pts.tolist()
        }
    return ptsSet
def structureExplorer(data, sizes, types):
    ''' Esplora ricorsivamente la struttura dati e applica l'engine ai DataFrame trovati. '''
    newData = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            newData[key] = engine(
                value,
                sizes.get(key, None), 
                types.get(key, None)
            )
        elif isinstance(value, dict):
            newData[key] = structureExplorer(
                value, 
                sizes.get(key, {}), 
                types.get(key, {})
            )
        else:
            newData[key] = value       
    return newData
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task1
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
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
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