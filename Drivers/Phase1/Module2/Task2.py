#%% Importing Libreries
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
#%% Defining Subroutines
def calibration(Config, imgpoints, shape):
    ''' Calibra la telecamera testando i vari modelli di distorsione forniti nel config. '''
    models_class = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2.Parameters.Models
    model_attrs = [m for m in dir(models_class) if not m.startswith("__")]
    model_attrs = sorted(model_attrs, key=lambda x: int(x.replace('Model', '')) if 'Model' in x else x)
    models = {
        m: getattr(models_class, m).value
        for m in model_attrs
    }
    keys = ('Ret', 'K', 'D', 'R', 'T')
    rows = Config.Settings.Pattern.patternRow
    cols = Config.Settings.Pattern.patternCol
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = objp * Config.Settings.Pattern.patternScale
    objpoints = [objp for _ in imgpoints]
    criteria = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2.Parameters.criteria
    base_flag = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2.Parameters.flag
    calib = pd.DataFrame(index=keys, columns=models.keys())
    img_size = (int(shape[1]), int(shape[0]))
    for modelName, modelVal in models.items():
        flag = modelVal | base_flag
        ret, K, D, R, T = cv.calibrateCamera(
            objpoints, imgpoints, img_size,
            None, None, criteria=criteria, flags=flag)
        calib[modelName] = [ret, K, D, R, T]
    return calib
def engine(Config, data, shapes, path):
    ''' Aggrega i record e lancia la calibrazione per ciascuna telecamera. '''
    merged = {}
    for subdict in data.values():
        for key in subdict:
            if key not in merged:
                merged[key] = []
            merged[key].extend(subdict[key])
    calib = {}
    for camera in merged.keys():
        check = []
        val = shapes[camera]
        for key in path:
            val = val[key] 
        check.extend([val[k] for k in data.keys()])
        if all(t == check[0] for t in check):
            calib[camera] = calibration(Config, merged[camera], check[0])
        else:
            raise ValueError(f'Dimensioni (shape) differenti trovate per i record della telecamera {camera}')
    return calib
def structureExplorer(Config, data, shapes, path, pbar):
    ''' Esplora ricorsivamente la struttura e applica l'engine quando trova le foglie con i punti. '''
    newData = {}
    for key, value in data.items():
        current = path + [key]
        if isinstance(value, dict):
            if all(isinstance(v, dict) and
                   all(isinstance(inner_v, list) for inner_v in v.values())
                   for v in value.values()):
                newData[key] = engine(Config, value, shapes, current)
            else:
                newData[key] = structureExplorer(Config, value, shapes, current, pbar)
        else:
            newData[key] = value
        pbar.update(1)
        pbar.set_postfix_str(colored(f"Dataset: {'/'.join(current)}", "magenta"))
    return newData
def count_nodes(d):
    ''' Semplice funzione ricorsiva per contare le iterazioni previste dal dizionario. '''
    count = 0
    for k, v in d.items():
        count += 1
        if isinstance(v, dict):
            count += count_nodes(v)
    return count
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ / 
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
        shapeRoot = (main_root /
                     Config.Paths.DataRoots.ResourcesRoot /
                     Config.Paths.DataRoots.StreamRoot /
                     Config.Paths.DataRoots.CaseStudyRoot() /
                     Config.Packages.Drivers.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.ShapeExt)   
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
            raise FileNotFoundError(f"Dati sorgenti mancanti.")
        if not shapeRoot.exists():
            raise FileNotFoundError(f"Metadati Shape mancanti.")
        dstRoot.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = pd.read_pickle(srcRoot)
            shapes = dict(pd.read_json(shapeRoot).loc[task_conf.Settings.Database])
            total_items = count_nodes(data)
            with tqdm(total=total_items, desc=colored('Mono Camera Calibration 🚀', 'magenta')) as pbar:
                result = structureExplorer(Config, data, shapes, [], pbar)
            with open(dstRoot, "wb") as f:
                pickle.dump(result, f) 
            print('.... Task2:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return