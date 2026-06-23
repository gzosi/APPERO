#%% Importing Libreries
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
#%% Defining Subroutines
def getMeanError(Config, imgpoints, calib):
    ''' Calcola l'errore medio di riproiezione per i parametri di calibrazione forniti. '''
    rows = Config.Settings.Pattern.patternRow
    cols = Config.Settings.Pattern.patternCol
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = objp * Config.Settings.Pattern.patternScale
    objpoints = [objp for _ in imgpoints]
    mean_error = 0  
    for i in range(len(objpoints)):
        test_points, _ = cv.projectPoints(
            objpoints[i], 
            calib['R'][i], calib['T'][i], calib['K'], calib['D']
        )   
        error = cv.norm(imgpoints[i], test_points, cv.NORM_L2) / len(test_points)  
        mean_error += error
    return mean_error / len(objpoints)
def engine(Config, calib, data):
    ''' Calcola gli errori medi per tutti i modelli di calibrazione della telecamera corrente. '''
    pts = {}
    for d in data.values():
        for key, lst in d.items():
            pts.setdefault(key, []).extend(lst)
    meanErrors_dict = {}
    for camera in pts.keys() & calib.keys():
        camera_calib = calib[camera]
        camera_pts = pts[camera]
        meanErrors_dict[camera] = {}
        for model in camera_calib.columns:
            meanErrors_dict[camera][model] = getMeanError(Config, camera_pts, camera_calib[model])
    meanErrors = pd.DataFrame.from_dict(meanErrors_dict, orient='index')
    return meanErrors
def structureExplorer(Config, calib, data, path, pbar):
    ''' Esplora ricorsivamente la struttura dati e processa i livelli terminali (record). '''
    newData = {}
    for key, value in data.items():
        current = path + [key]
        if isinstance(value, dict):
            if all(isinstance(v, dict) and
                   all(isinstance(inner_v, list) for inner_v in v.values())
                   for v in value.values()):
                newData[key] = engine(Config, calib[key], value)
            else:
                newData[key] = structureExplorer(Config, calib[key], value, current, pbar)
        else:
            newData[key] = value
        pbar.update(1)
        pbar.set_postfix_str(colored(f"Dataset: {'/'.join(current)}", "magenta"))
    return newData
def count_nodes(d):
    ''' Funzione ricorsiva per contare i nodi dell'albero e settare la progress bar. '''
    count = 0
    for k, v in d.items():
        count += 1
        if isinstance(v, dict):
            count += count_nodes(v)
    return count
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task3
    if task_conf.General.Activation is True:
        print('.... Task3:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        ptsRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task1.MetaData.OutputExt)      
        calibRoot = (main_root /
                     Config.Paths.DataRoots.ResourcesRoot /
                     Config.Paths.DataRoots.StreamRoot /
                     Config.Paths.DataRoots.CaseStudyRoot() /
                     Config.Packages.Drivers.__name__ /
                     Config.Packages.Drivers.Phases.Phase1.__name__ /
                     Config.Packages.Drivers.Phases.Phase1.Modules.Module2.__name__ /
                     Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2.__name__ /
                     Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2.MetaData.OutputExt)   
        dstRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.__name__ /
                   task_conf.__name__ /
                   task_conf.MetaData.OutputExt)
        if not ptsRoot.exists():
            raise FileNotFoundError(f"Dati dei punti mancanti!")
        if not calibRoot.exists():
            raise FileNotFoundError(f"Dati di calibrazione mancanti!")
        dstRoot.parent.mkdir(parents=True, exist_ok=True)
        try:
            calib = pd.read_pickle(calibRoot)
            pts = pd.read_pickle(ptsRoot)
            total_items = count_nodes(pts)
            with tqdm(total=total_items, desc=colored('Evaluating Reprojection Errors 🚀', 'magenta')) as pbar:
                results = structureExplorer(Config, calib, pts, [], pbar)    
            with open(dstRoot, "wb") as f:
                pickle.dump(results, f)
            print('.... Task3:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task3:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task3:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task3 Switch (on/off) ❌')
    return