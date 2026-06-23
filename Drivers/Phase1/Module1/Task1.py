#%% Importing Libreries
import h5py
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from termcolor import colored
#%% Defining Subroutines
#%% Defining Subroutines
def orientationAdjuster(kps):  
    ''' Questa funzione serve a regolare l'orientamento dei keypoints. '''
    tester = kps.reshape(-1, 2)
    X, Y = tester[-1] - tester[0]
    if X > 0 and Y > 0:
        kps = kps
    elif X < 0 and Y < 0:
        kps = np.flip(kps, axis=0)
    else:
        # raise ValueError('Qualcosa non va nell\'orientamento dei corner')
        return None
    return kps
def Engine(Config, camera_names, camera_groups, dataset_name=""): 
    ''' Trova i corner della scacchiera e applica il sub-pixel refinement. '''
    task_params = Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task1.Parameters
    patternSize = Config.Settings.Pattern.patternSize
    winSize = task_params.winSize
    zeroZone = task_params.zeroZone
    criteria = task_params.criteria
    flagA = task_params.Flags.typeA
    flagB = task_params.Flags.typeB
    subpixA = task_params.Subpix.typeA
    subpixB = task_params.Subpix.typeB
    common_files = sorted(set.intersection(*(set(group.keys()) for group in camera_groups)))
    data = pd.DataFrame(columns=camera_names)
    for id, file_name in enumerate(tqdm(common_files, desc=colored(f'Processing: {dataset_name} 🚀', 'magenta'), leave=True)):
        for cam_name, group in zip(camera_names, camera_groups):
            img = group[file_name][()]
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            points, ptsType = None, None
            success, points = cv.findChessboardCorners(gray, patternSize, None, flags=flagA)
            if success:
                if subpixA:
                    points = cv.cornerSubPix(gray, points, winSize, zeroZone, criteria)
                points = orientationAdjuster(points)
                ptsType = 'typeA'
            else:
                success, points = cv.findChessboardCornersSB(gray, patternSize, None, flags=flagB)
                if success:
                    if subpixB:
                        points = cv.cornerSubPix(gray, points, winSize, zeroZone, criteria)
                    points = orientationAdjuster(points)
                    ptsType = 'typeB'
            data.loc[id, cam_name] = [points, ptsType]
    return data
def exploreStructure(Config, camera_names, camera_groups, path=""):
    ''' Esplora ricorsivamente la struttura del file HDF5 trovando i path comuni. '''
    structure = {}
    common_keys = set.intersection(*(set(group.keys()) for group in camera_groups))
    for key in sorted(common_keys):
        items = [group[key] for group in camera_groups]
        current_path = f"{path}/{key}" if path else key
        if all(isinstance(item, h5py.Group) for item in items):
            if all(all(isinstance(subitem, h5py.Dataset) for subitem in item.values()) for item in items):
                structure[key] = Engine(Config, camera_names, items, dataset_name=current_path)
            else:
                structure[key] = exploreStructure(Config, camera_names, items, current_path)
    return structure
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored( 'Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ / 
                   Config.Packages.Drivers.Phases.Phase0.__name__ /
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)
        dstRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot / 
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.__name__ /
                   task_conf.__name__ /
                   task_conf.MetaData.OutputExt) 
        if not dstRoot.parent.exists():
            raise FileNotFoundError(f"Cartella di destinazione non trovata!")
        database = task_conf.Settings.Database
        try:
            with h5py.File(srcRoot, 'r') as f:
                cameras = {name: f[name][database] for name in f if database in f[name]}
                camera_names = list(cameras.keys())
                camera_groups = list(cameras.values())
                structure = exploreStructure(Config, camera_names, camera_groups)
            with open(dstRoot, 'wb') as f:
                pickle.dump(structure, f)
            print('.... Task1:', colored( 'Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored( f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
        
    return