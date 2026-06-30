#%% Importing Libreries
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import torch, h5py, pickle
from scipy.optimize import linear_sum_assignment
from kornia.feature import LoFTR 
from tqdm import tqdm
from termcolor import colored
#%% Defining Subroutines
def padOrigin(img, shape, origin):
    result = np.zeros((int(shape[0]), int(shape[1])), dtype=img.dtype)
    y_start, x_start = int(origin[1]), int(origin[0])
    y_end = y_start + img.shape[0]
    x_end = x_start + img.shape[1]
    result[y_start:y_end, x_start:x_end] = img
    return result
def matchEngine(epiLimit, confLimit, img1, img2, matcher, device, calib):
    ''' Trova i match con LoFTR, filtra tramite geometria epipolare. '''
    tensor1 = torch.from_numpy(img1).float()[None, None].to(device) / 255.0
    tensor2 = torch.from_numpy(img2).float()[None, None].to(device) / 255.0
    with torch.inference_mode(): 
        batch = matcher({'image0': tensor1, 'image1': tensor2})
    conf = batch['confidence'].cpu().numpy()
    pts1_raw = batch['keypoints0'].cpu().numpy()
    pts2_raw = batch['keypoints1'].cpu().numpy()
    pts1_h = np.hstack([pts1_raw, np.ones((pts1_raw.shape[0], 1))]) 
    pts2_h = np.hstack([pts2_raw, np.ones((pts2_raw.shape[0], 1))]) 
    F = calib['F']
    errors = np.sum(pts2_h @ F * pts1_h, axis=1) # x2^T * F * x1
    Fp1 = F @ pts1_h.T
    Fp2 = F.T @ pts2_h.T
    sampson_error = (errors**2) / (Fp1[0,:]**2 + Fp1[1,:]**2 + Fp2[0,:]**2 + Fp2[1,:]**2)
    mask = (sampson_error < epiLimit) & (conf > confLimit)
    valid_pts1 = pts1_raw[mask]
    valid_pts2 = pts2_raw[mask]
    return valid_pts1, valid_pts2
def inMasksPts(pts, masks):
    if not masks or len(masks) == 0:
        return []
    if pts is None or len(pts) == 0:
        return [np.array([], dtype=int) for _ in masks]
    h, w = masks[0].shape
    pts_int = np.round(pts).astype(int)
    if pts_int.ndim == 1 or pts_int.shape[1] < 2:
         return [np.array([], dtype=int) for _ in masks]
    x = np.clip(pts_int[:, 0], 0, w - 1)
    y = np.clip(pts_int[:, 1], 0, h - 1)
    masks_array = np.asarray(masks)
    hits = masks_array[:, y, x] > 0
    return [np.where(row)[0] for row in hits]
def split_masks(mask):
    """
    Splits a binary image into a list of individual binary masks.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    num_labels, labels = cv.connectedComponents(mask)
    masks = []
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8)
        masks.append(mask)        
    return masks
def getConnection(simLimit, idx1, idx2):
    if not idx1 or not idx2:
        return []
    sets1 = [set(a) for a in idx1]
    sets2 = [set(b) for b in idx2]
    connection = np.array([
        [(2 * len(sa & sb) / (len(sa) + len(sb))) if sa and sb else 0.0
         for sb in sets2]
        for sa in sets1])
    if connection.ndim != 2 or connection.size == 0:
        return []
    rowIdx, colIdx = linear_sum_assignment(-connection)
    bestCons = []
    for i, j in zip(rowIdx, colIdx):
        score = connection[i, j]
        if score >= simLimit: bestCons.append([i, j, score])
    return bestCons
#%% Defining Main Function
def main(Config):    
    task_conf = Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task3:', colored('Running ℹ️', 'cyan'))
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
            Config.Packages.Drivers.Phases.Phase4.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task2.MetaData.OutputExt)
        calibRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ /
            Config.Packages.Drivers.Phases.Phase1.__name__ /
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ /
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt)
        shapeRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot /
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase0.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.__name__)
        objRoot = (main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot /
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task1.MetaData.OutputExt)
        settings = task_conf.Settings
        data = pd.read_pickle(objRoot)
        origins_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt)
        shapes_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.ShapeExt)
        cameras = list(origins_data.keys())
        shape1, shape2 = [
            shapes_data[camera][settings.Ref.Database][settings.Ref.Dataset][settings.Ref.Record]
            for camera in shapes_data.keys()]
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset]  
        matcher = LoFTR(pretrained='outdoor').eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        matcher = matcher.to(device)
        try:
            with h5py.File(srcRoot, 'r') as f:
                cameras = list(f.keys())
                cam1, cam2 = cameras[0], cameras[1]
                datasets = data.keys()
                aggregate = {}
                for dataset in datasets:
                    group1 = f[cam1][settings.Src.Database][dataset]
                    group2 = f[cam2][settings.Src.Database][dataset]
                    origin1 = origins_data[cam1][settings.Src.Database][dataset]
                    origin2 = origins_data[cam2][settings.Src.Database][dataset]
                    keys = data[dataset].keys()
                    aggregate[dataset] = {}
                    for key in tqdm(keys, total=len(keys), desc=colored(f'Object Matching 🚀', 'magenta'), ncols=100):  
                        img1 = padOrigin(group1[key][:].astype(np.uint8), shape1, origin1)
                        img2 = padOrigin(group2[key][:].astype(np.uint8), shape2, origin2)
                        masks1 = split_masks(data[dataset][key][cam1])
                        masks2 = split_masks(data[dataset][key][cam2])
                        kpts2a, kpts1a = matchEngine(
                            settings.EpiLimit, settings.ConfLimit,
                            img1, img2, matcher, device, 
                            calib[(cam1, cam2)][settings.Calib.Model])
                        kpts2b, kpts1b = matchEngine(
                            settings.EpiLimit, settings.ConfLimit,
                            img2, img1, matcher, device, 
                            calib[(cam2, cam1)][settings.Calib.Model])
                        kpts1 = np.concatenate((kpts1a, kpts1b)) if len(kpts1a) else kpts1b
                        kpts2 = np.concatenate((kpts2a, kpts2b)) if len(kpts2a) else kpts2b
                        inMasks1, inMasks2 = inMasksPts(kpts1, masks1), inMasksPts(kpts2, masks2)
                        connections = getConnection(settings.SimLimit, inMasks1, inMasks2)
                        rows = []
                        for _, (i, j, _) in enumerate(connections):
                            rows.append({cam1: masks1[i], cam2: masks2[j]})
                        aggregate[dataset][key] = pd.DataFrame(rows)
            pickle.dump(aggregate, open(dstRoot, 'wb'))
            if torch.cuda.is_available(): torch.cuda.empty_cache()       
            print('.... Task3:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task3:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task3:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task3 Switch (on/off) ❌')
    return 