#%% Importing Libraries
from pathlib import Path
import pandas as pd
import h5py
import numpy as np
import cv2 as cv
from tqdm import tqdm 
from scipy.spatial import cKDTree
from termcolor import colored
#%% Defining Subroutines
def padOrigin(img, shape, origin):
    ''' Ripristina l'immagine croppata alla sua dimensione e posizione originale sul sensore. '''
    result = np.zeros((int(shape[0]), int(shape[1])), dtype=img.dtype)
    y_start, x_start = int(origin[1]), int(origin[0])
    y_end = y_start + img.shape[0]
    x_end = x_start + img.shape[1]
    result[y_start:y_end, x_start:x_end] = img
    return result
def track(Config, currImg, nexImg, currKpts, nextKpts):
    distLimit = Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task2.Tracking.distLimit
    if len(currKpts) == 0 or len(nextKpts) == 0:
        return np.array([]), np.array([], dtype=bool)    
    lkParam = Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task2.Tracking.LKparam
    p1 = currKpts.astype(np.float32)
    p2, status, _ = cv.calcOpticalFlowPyrLK(currImg, nexImg, p1, None, **lkParam)
    flow = status.reshape(-1) == 1
    tree = cKDTree(nextKpts)
    dists, idxs = tree.query(p2)
    valid_match = flow & (dists < distLimit)
    return idxs, valid_match
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
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
            Config.Packages.Drivers.Phases.Phase2.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.__name__ /
            task_conf.__name__ /
            task_conf.MetaData.OutputExt)
        steroRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase2.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1.MetaData.OutputExt)
        shapeRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot /
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase0.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        shapes_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.ShapeExt)
        origins_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt)
        cameras = list(shapes_data.keys())
        shape1, shape2 = [
            shapes_data[camera][settings.Ref.Database][settings.Ref.Dataset][settings.Ref.Record]
            for camera in cameras]
        origin1, origin2 = [
            origins_data[camera][settings.Src.Database][settings.Src.Dataset]
            for camera in cameras]
        stereo = pd.read_pickle(steroRoot)
        try:
            links = []
            with h5py.File(srcRoot, 'r') as f:
                group1, group2 = [
                    f[camera][settings.Src.Database][settings.Src.Dataset]
                    for camera in cameras]
                keys = sorted(group1.keys())
                for idx, key in tqdm(enumerate(keys), desc=colored('Stereo-Tempo Consistent Tracking,🚀', 'magenta'), ncols=100):
                    nextKey = keys[idx+1] if idx + 1 < len(keys) else None
                    if key not in stereo.index or nextKey not in stereo.index: continue
                    currImg1 = padOrigin(group1[key][:].astype(np.uint8), shape1, origin1)
                    currImg2 = padOrigin(group2[key][:].astype(np.uint8), shape2, origin2)
                    nextImg1 = padOrigin(group1[nextKey][:].astype(np.uint8), shape1, origin1)
                    nextImg2 = padOrigin(group2[nextKey][:].astype(np.uint8), shape2, origin2)
                    currPts1, currPts2 = stereo.loc[key, cameras[0]], stereo.loc[key, cameras[1]]
                    nextPts1, nextPts2 = stereo.loc[nextKey, cameras[0]], stereo.loc[nextKey, cameras[1]]
                    idxs1, mask1 = track(Config, currImg1, nextImg1, np.array(currPts1), np.array(nextPts1))
                    idxs2, mask2 = track(Config, currImg2, nextImg2, np.array(currPts2), np.array(nextPts2))
                    mask = mask1 & mask2 & (idxs1 == idxs2)
                    currIndices, nextIndices  = np.where(mask)[0], idxs1[mask] 
                    if len(currIndices) > 0:
                        links.append({
                            'Keys': (key, nextKey),  
                            'Phase': stereo.loc[key, 'Phase'],
                            'Current': currIndices,  
                            'Next': nextIndices})
            if len(links) > 0:
                data = pd.DataFrame(links).set_index('Keys')
                data.to_pickle(dstRoot) 
            else: raise ValueError('No data collected')
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return