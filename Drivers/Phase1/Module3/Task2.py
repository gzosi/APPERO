#%% Importing Libreries
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from termcolor import colored
from tqdm import tqdm
#%% Defining Subroutines
def calibration(Config, imgpoints, monoCalib, shape):
    ''' Calibra il rig stereo usando i dati forniti e le calibrazioni mono precedenti. '''
    models_class = Config.Packages.Drivers.Phases.Phase1.Modules.Module2.Tasks.Task2.Parameters.Models
    model_attrs = [m for m in dir(models_class) if not m.startswith("__")]
    model_attrs = sorted(model_attrs, key=lambda x: int(x.replace('Model', '')) if 'Model' in x else x)
    models = {m: getattr(models_class, m).value for m in model_attrs}
    keys = ('Ret', 'K1', 'D1', 'K2', 'D2', 'R', 'T', 'E', 'F', 'P1', 'P2')
    rows = Config.Settings.Pattern.patternRow
    cols = Config.Settings.Pattern.patternCol
    imgpoints1 = imgpoints[0]
    imgpoints2 = imgpoints[1]
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp = objp * Config.Settings.Pattern.patternScale
    objpoints = [objp for _ in imgpoints1]
    if shape[0] != shape[1]:
        print(colored('⚠️⚠️ Attenzione: Risoluzione camere differente rilevata! ⚠️⚠️', 'yellow'))
    criteria = Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.Parameters.criteria
    base_flag = Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.Parameters.flag
    stereoCalib = pd.DataFrame(index=keys, columns=models.keys())
    img_size = (int(shape[0][1]), int(shape[0][0]))
    for modelName, modelVal in models.items():
        flag = modelVal | base_flag
        K1_init = monoCalib[0][modelName]['K']
        D1_init = monoCalib[0][modelName]['D']
        K2_init = monoCalib[1][modelName]['K']
        D2_init = monoCalib[1][modelName]['D']
        ret, K1, D1, K2, D2, R, T, E, F = cv.stereoCalibrate(
            objpoints, imgpoints1, imgpoints2,
            K1_init, D1_init,
            K2_init, D2_init,
            img_size, criteria=criteria, flags=flag
        )
        P1 = np.dot(K1, np.concatenate([np.eye(3), np.zeros((3, 1))], axis=-1))
        P2 = np.dot(K2, np.concatenate([R, T], axis=-1))
        stereoCalib[modelName] = [ret, K1, D1, K2, D2, R, T, E, F, P1, P2]
    return stereoCalib
def engine(Config, data, monoCalib, shapes, path):
    ''' Aggrega i dati e chiama l'ottimizzatore stereo per ciascuna coppia. '''
    steroCalib, merged = {}, {}
    for k in path: 
        monoCalib = monoCalib[k]
    for subdict in data.values():  
        for k1, inner_dict in subdict.items(): 
            if k1 not in merged:
                merged[k1] = {}
            for k2, lst in inner_dict.items():  
                if k2 not in merged[k1]:
                    merged[k1][k2] = []
                merged[k1][k2].extend(lst)
    for pair in merged.keys():
        check1, check2 = [], []
        val1, val2 = shapes[pair[0]], shapes[pair[1]]
        for key in path:
            val1, val2 = val1[key], val2[key]
        check1.extend([val1[k] for k in data.keys()])
        check2.extend([val2[k] for k in data.keys()])
        if all(t == check1[0] for t in check1) and all(t == check2[0] for t in check2):
            steroCalib[pair] = calibration(
                Config,
                (merged[pair][pair[0]], merged[pair][pair[1]]),
                (monoCalib[pair[0]], monoCalib[pair[1]]),
                (check1[0], check2[0])
            )
        else:
            raise ValueError(f'Dimensioni (shape) differenti trovate per la coppia {pair}')
    return steroCalib
def structureExplorer(Config, data, monoCalib, shapes, path, pbar):
    ''' Esplora ricorsivamente la struttura dati per trovare i punti da calibrare. '''
    newData = {}
    for key, value in data.items():
        current = path + [key]
        if isinstance(value, dict):
            if all(isinstance(v, dict) and
                   all(isinstance(inner_v, dict) and
                       all(isinstance(i, list) for i in inner_v.values())
                       for inner_v in v.values())
                   for v in value.values()):
                newData[key] = engine(Config, value, monoCalib, shapes, current)
            else:
                newData[key] = structureExplorer(Config, value, monoCalib, shapes, current, pbar)
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
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task1.MetaData.OutputExt)
        shapeRoot = (main_root /
                     Config.Paths.DataRoots.ResourcesRoot /
                     Config.Paths.DataRoots.StreamRoot /
                     Config.Paths.DataRoots.CaseStudyRoot() /
                     Config.Packages.Drivers.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__ /
                     Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.ShapeExt)
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
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
                   task_conf.__name__ /
                   task_conf.MetaData.OutputExt)
        if not srcRoot.exists():
            raise FileNotFoundError(f"Dati non trovati!")
        if not calibRoot.exists():
            raise FileNotFoundError(f"Calibrazioni Mono non trovate!")
        dstRoot.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = pd.read_pickle(srcRoot)
            monoCalib = pd.read_pickle(calibRoot)
            shapes = dict(pd.read_json(shapeRoot).loc[task_conf.Settings.Database])
            total_items = count_nodes(data)
            with tqdm(total=total_items, desc=colored('Stereo Camera Calibration 🚀', 'magenta')) as pbar:
                result = structureExplorer(Config, data, monoCalib, shapes, [], pbar)
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