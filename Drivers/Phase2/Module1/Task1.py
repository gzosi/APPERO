#%% Importing Libreries
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from kornia.feature import LoFTR 
import h5py
from pathlib import Path
from tqdm import tqdm
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
#%% Defining Main Function
def main(Config):  
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1
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
        calib_data = pd.read_pickle(calibRoot)[settings.Calib.Dataset]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        matcher = LoFTR(pretrained='outdoor').eval().to(device)
        try:
            with h5py.File(srcRoot, 'r') as f:
                group1, group2 = [
                    f[camera][settings.Src.Database][settings.Src.Dataset]
                    for camera in cameras]
                proces = [
                    k for k in sorted(group1.keys())
                    if any(lower <= (int(k) % ppr) <= upper 
                    for lower, upper in settings.Syncrony.Bounds)]
                data = pd.DataFrame(columns=cameras, index=proces)
                data['Phase'] = [int(k) % ppr for k in proces]
                for key in tqdm(proces, desc=colored('Surface Marker Extraction 🚀', 'magenta'), ncols=100):
                    img1 = padOrigin(group1[key][:].astype(np.uint8), shape1, origin1)
                    img2 = padOrigin(group2[key][:].astype(np.uint8), shape2, origin2)
                    kpts1a, kpts2a = matchEngine(
                        settings.EpiLimit, settings.ConfLimit,
                        img1, img2, matcher, device,
                        calib_data[(cameras[0], cameras[1])][settings.Calib.Model])  
                    kpts2b, kpts1b = matchEngine(
                        settings.EpiLimit, settings.ConfLimit,
                        img2, img1, matcher, device,
                        calib_data[(cameras[1], cameras[0])][settings.Calib.Model])  
                    kpts1 = np.concatenate((kpts1a, kpts1b)) if len(kpts1a) else kpts1b
                    kpts2 = np.concatenate((kpts2a, kpts2b)) if len(kpts2a) else kpts2b
                    data.at[key, cameras[0]] = kpts1.tolist()
                    data.at[key, cameras[1]] = kpts2.tolist()
            data.to_pickle(dstRoot)
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return