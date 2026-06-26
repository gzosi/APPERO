#%% Importing Libreries
from pathlib import Path
import h5py
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from kornia.feature import LoFTR 
from termcolor import colored
import pickle
#%% Defining Subroutines
def getColuds(cloud, cavities):
    if cloud is None or not cavities:
        return []    
    results = []
    for m in cavities:
        results.append(cloud & m)
    return results
def padOrigin(img, shape, origin):
    result = np.zeros((int(shape[0]), int(shape[1])), dtype=img.dtype)
    y_start, x_start = int(origin[1]), int(origin[0])
    y_end = y_start + img.shape[0]
    x_end = x_start + img.shape[1]
    result[y_start:y_end, x_start:x_end] = img
    return result
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
def applyTexturedMask(img, mask):
    ''' 
    Applica la maschera all'immagine grayscale. 
    Mantiene il background a 0 e forza i pixel neri (0) DENTRO la maschera al valore 1.
    '''
    bool_mask = mask > 0
    result = np.zeros_like(img)
    result[bool_mask] = img[bool_mask]
    zeros_in_mask = bool_mask & (img == 0)
    result[zeros_in_mask] = 1
    return result
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
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
            Config.Packages.Drivers.Phases.Phase3.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
            task_conf.__name__)
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
            Config.Packages.Drivers.Phases.Phase3.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task1.__name__)
        settings = task_conf.Settings
        shapes_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.ShapeExt)
        origins_data = pd.read_json(shapeRoot / Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2.MetaData.OriginExt)
        shape1, shape2 = [
            shapes_data[camera][settings.Ref.Database][settings.Ref.Dataset][settings.Ref.Record]
            for camera in shapes_data.keys()]
        origin1, origin2 = [
            origins_data[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Foreground]
            for camera in origins_data.keys()]
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset]   
        keys = [
            (''.join(filter(str.isdigit, f.name)), f.name)
            for f in objRoot.iterdir()
            if f.is_file() and any(c.isdigit() for c in f.name)]
        matcher = LoFTR(pretrained='outdoor').eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        matcher = matcher.to(device)
        try:
            with h5py.File(srcRoot, 'r') as f:
                cameras = list(f.keys())
                group1, group2 = [
                    f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Foreground]
                    for camera in cameras]              
                for (key, name) in tqdm(keys, total=len(keys),  desc=colored(f'Object Matching 🚀', 'magenta'), ncols=100):  
                    img1 = padOrigin(group1[key][:].astype(np.uint8), shape1, origin1)
                    img2 = padOrigin(group2[key][:].astype(np.uint8), shape2, origin2)
                    (cavities1, cloud1, _), (cavities2, cloud2, _) = [
                        pd.read_pickle(objRoot / name)[camera] 
                        for camera in cameras]
                    padCavities1 = [padOrigin(cavity1, shape1, origin1) for cavity1 in cavities1]
                    padCavities2 = [padOrigin(cavity2, shape2, origin2) for cavity2 in cavities2]
                    padClouds1 = [
                        padOrigin(c, shape1, origin1) for c in 
                        getColuds(cloud1, cavities1)]
                    padClouds2 = [
                        padOrigin(c, shape2, origin2) for c in 
                        getColuds(cloud2, cavities2)]
                    kpts2a, kpts1a = matchEngine(
                        settings.EpiLimit, settings.ConfLimit,
                        img1, img2, matcher, device, 
                        calib[(cameras[0], cameras[1])][settings.Calib.Model])
                    kpts2b, kpts1b = matchEngine(
                        settings.EpiLimit, settings.ConfLimit,
                        img2, img1, matcher, device, 
                        calib[(cameras[1], cameras[0])][settings.Calib.Model])
                    kpts1 = np.concatenate((kpts1a, kpts1b)) if len(kpts1a) else kpts1b
                    kpts2 = np.concatenate((kpts2a, kpts2b)) if len(kpts2a) else kpts2b
                    inCavityIdx1, inCloudIdx1 = inMasksPts(kpts1, padCavities1), inMasksPts(kpts1, padClouds1)
                    inCavityIdx2, inCloudIdx2 = inMasksPts(kpts2, padCavities2), inMasksPts(kpts2, padClouds2)
                    connections = getConnection(settings.SimLimit, inCavityIdx1, inCavityIdx2)
                    data = pd.DataFrame(index=range(len(connections)), columns=cameras)
                    for k, (i, j, _) in enumerate(connections):
                        commonCavity = list(set(inCavityIdx1[i]) & set(inCavityIdx2[j]))
                        commonCloud = list(set(inCloudIdx1[i]) & set(inCloudIdx2[j]))
                        cavityMatched1, cavityMatched2 = kpts1[commonCavity], kpts2[commonCavity]
                        cloudMatched1, cloudMatched2 = kpts1[commonCloud], kpts2[commonCloud]
                        texCavity1 = applyTexturedMask(img1, padCavities1[i])
                        texCloud1  = applyTexturedMask(img1, padClouds1[i])
                        texCavity2 = applyTexturedMask(img2, padCavities2[j])
                        texCloud2  = applyTexturedMask(img2, padClouds2[j])
                        data.at[k, cameras[0]] = [
                            texCavity1, texCloud1,
                            cavityMatched1, cloudMatched1]
                        data.at[k, cameras[1]] = [
                            texCavity2, texCloud2, 
                            cavityMatched2, cloudMatched2]
                    dst = (dstRoot / f'{task_conf.MetaData.OutputName}_{key}{task_conf.MetaData.OutputExt}')
                    pickle.dump(data, open(dst, 'wb'))
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            print('.... Task2:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return