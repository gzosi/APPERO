#%% Importing Libreries
import numpy as np
import pandas as pd
from pathlib import Path
from termcolor import colored
#%% Defining Subroutines
def getConnections(Config):
    ''' Ottiene le connessioni attese tra i nodi (vertici) del pattern a scacchiera. '''
    rows = Config.Settings.Pattern.patternRow
    cols = Config.Settings.Pattern.patternCol
    data = []
    for i in range(cols): 
        for j in range(rows - 1): 
            start = j + (rows * i)
            end = start + 1
            data.append({"edge": (start, end), "type": "row", "index": i})
    for i in range(rows): 
        for j in range(cols - 1):  
            start = (j * rows) + i
            end = start + rows
            data.append({"edge": (start, end), "type": "col", "index": i})
    return pd.DataFrame(data)
def getDeviations(data, connections):
    ''' Calcola le deviazioni geometriche dei nodi per scartare estrazioni deformate. '''
    deviations = pd.DataFrame(index=data.index, columns=data.columns)
    for camera in data.columns:
        ptsArray = data[camera].map(lambda x: x[0] if isinstance(x, list) and x[0] is not None else None).tolist()
        for i, pts in enumerate(ptsArray):
            if pts is None or len(pts) == 0:
                deviations[camera].at[i] = np.nan
                continue
            pts = pts.reshape(pts.shape[0], pts.shape[2])
            distances = [np.linalg.norm(pts[start] - pts[end]) for (start, end) in connections['edge']]
            types = connections['type'].tolist()
            log = pd.DataFrame({'distance': distances, 'type': types})
            means = log.groupby('type')['distance'].transform('mean')
            log['deviation'] = abs(log['distance'] - means) / means * 100
            deviations[camera].at[i] = log['deviation'].max()
    return deviations
def engine(Config, data):
    ''' Filtra i dati: se la deviazione supera il limite di guardia, la entry diventa NaN. '''
    limit = Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task2.Settings.limit
    connections = getConnections(Config)
    deviations = getDeviations(data, connections)
    mask = deviations > limit
    filtered = data.mask(mask)
    return filtered
def structureExplorer(Config, data):
    ''' Esplora ricorsivamente i dizionari e applica l'engine ai DataFrame trovati. '''
    newData = {}
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            newData[key] = engine(Config, value)
        elif isinstance(value, dict):
            newData[key] = structureExplorer(Config, value)
        else:
            newData[key] = value
    return newData
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ / 
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.Tasks.Task1.MetaData.OutputExt)
        dstRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.__name__ /
                   Config.Packages.Drivers.Phases.Phase1.Modules.Module1.__name__ /
                   task_conf.__name__ /
                   task_conf.MetaData.OutputExt)
        if not srcRoot.exists():
            raise FileNotFoundError(f"File sorgente non trovato!")
        dstRoot.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = pd.read_pickle(srcRoot)
            newData = structureExplorer(Config, data)
            pd.to_pickle(newData, dstRoot)
            print('.... Task2:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
        
    return