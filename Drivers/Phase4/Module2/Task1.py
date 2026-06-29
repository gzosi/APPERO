#%% Importing Libreries
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
from tqdm import tqdm
from termcolor import colored 
#%% Defining Subroutines
def surfaceCarver(masks: list, calib: dict, pose: np.ndarray, config) -> tuple:
    """Intaglia la Superficie dell'elica usando le maschere di occlusione."""
    filled = []
    for i, mask in enumerate(masks):
        imgH, imgW = mask.shape
        ys, xs = np.where(mask > 0)
        points = np.stack([xs, ys], axis=-1)  
        zero = np.zeros_like(mask, dtype=np.uint8)
        if points.size > 0:
            K, D = calib.get(f"K{i+1}"), calib.get(f"D{i+1}")
            undistorted = cv.undistortPoints(points.astype(np.float32).reshape(-1, 1, 2), K, D, None, K).reshape(-1, 2)
            undistorted = np.round(undistorted).astype(int)
            valid = ((undistorted[:, 0] >= 0) & (undistorted[:, 0] < imgW) & (undistorted[:, 1] >= 0) & (undistorted[:, 1] < imgH))
            undistorted = undistorted[valid]
            zero[undistorted[:, 1], undistorted[:, 0]] = 1
        poseH = np.hstack([pose, np.ones((pose.shape[0], 1))])
        P = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1)))) if i == 0 else calib['K2'] @ np.hstack((calib['R'], calib['T']))
        uvs = P @ poseH.T
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        good = (uvs[0, :] >= 0) & (uvs[0, :] < imgW) & (uvs[1, :] >= 0) & (uvs[1, :] < imgH)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        fill[indices] = zero[sub_uvs[1, :], sub_uvs[0, :]]
        filled.append(fill) 
    occupancy = np.sum(np.vstack(filled), axis=0)
    good_indices = np.where(occupancy >= config.occupancyLimit)[0]
    return good_indices, pose[good_indices]
def getFingerprint(df, blades):
    """Crea un dizionario con la firma di ogni pala: {blade_id: [centroid_x, y, z, area]}"""
    fingerprints = {}
    for b in range(1, blades + 1):
        subset = df[df['Blade'] == b]
        if subset.empty:
            fingerprints[b] = np.array([0.0, 0.0, 0.0, 0.0])
            continue
        all_pts = np.vstack(subset['Damage'].values)
        centroid = np.mean(all_pts, axis=0)
        total_area = subset['Area'].sum()
        fingerprints[b] = np.array([*centroid, total_area * 0.01]) 
    return fingerprints
#%% Defining Main Function
def main(Config): 
    task_conf = Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        poseRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.MetaData.OutputExt)
        objRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task2.__name__ /
            Config.Packages.Drivers.Phases.Phase4.Modules.Module1.Tasks.Task2.MetaData.OutputExt)
        calibRoot = (
            main_root /
            Config.Paths.DataRoots.ResourcesRoot /
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ / 
            Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt)
        dstRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.__name__ / 
            Config.Packages.Drivers.Phases.Phase4.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        poses, areas = [pd.read_pickle(poseRoot)[key] for key in ['single', 'areas']]
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
        objects = pd.read_pickle(objRoot)
        datasets = list(objects.keys())
        try:
            data = {}
            for dataset in datasets:
                results = []
                for key in tqdm(list(objects[dataset].keys()), desc=colored('Surface Carving 🚀', 'magenta'), ncols=100): 
                    phase = int(key) % ppr 
                    pose, masks = poses[phase], objects[dataset][key]
                    pts, ids = set(), set()
                    for _, (mask1, mask2) in masks.iterrows():
                        id_res, damage = surfaceCarver([mask1, mask2], calib, pose, settings.Carver)
                        if damage is not None and len(damage): 
                            pts.update(map(tuple, damage))
                        if id_res is not None and id_res.size: 
                            ids.update(id_res.tolist())
                    results.append({
                        'key': key,
                        'Pose': pose,
                        'Damage': np.array(list(pts)),
                        'Area': sum(areas[i] for i in ids) if ids else 0.0,
                        'Blade': int((int(key) % ppr) // (ppr / blades)) + 1})
                data[dataset] = pd.DataFrame(results).set_index('key')
            for i in range(1, len(datasets)):
                prev_ds = datasets[i-1]
                curr_ds = datasets[i]
                f_prev = getFingerprint(data[prev_ds], blades)
                f_curr = getFingerprint(data[curr_ds], blades)
                best_shift = 0
                min_cost = float('inf')
                for shift in range(blades):
                    current_cost = 0
                    for b in range(1, blades + 1):
                        target = f_prev[b]
                        curr_b = ((b - 1 + shift) % blades) + 1
                        source = f_curr[curr_b]
                        current_cost += np.linalg.norm(target - source)
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_shift = shift
                if best_shift != 0:
                    data[curr_ds]['Blade'] = ((data[curr_ds]['Blade'] - 1 + best_shift) % blades) + 1
            pickle.dump(data, open(dstRoot, 'wb'))
            print('.... Task2:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return