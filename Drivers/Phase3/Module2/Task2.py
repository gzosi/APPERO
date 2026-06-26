#%% Importing Libreries
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
import pickle
from termcolor import colored
def getMetrix(cloud1, cloud2):
    """
    Takes two cavitation images as input, calculates Average Intensity (Saturation), 
    Skewness, and Entropy, and returns the averaged normalized values [Sat, Skw, Ent].
    
    Legend:
    Sat: 100 = white (dense), 0 = dark (sparse)
    Skw: 100 = high density distribution, 1 = low
    Ent: 100 = high instability/chaos, 1 = stable
    """
    def engine(img):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        pixels = img[img > 5]
        if len(pixels) == 0:
            return 0.0, 2.0, 0.0  
        area = len(pixels)
        # 1. Average Intensity (Closeness to white 255)
        raw_sat = np.mean(pixels)
        # 2. Skewness (Density Distribution)
        mean_val = np.mean(pixels)
        std_dev = np.std(pixels)
        raw_skw = np.mean(((pixels - mean_val) / std_dev) ** 3) if std_dev > 0 else 0.0
        # 3. Entropy (Chaos)
        hist, _ = np.histogram(pixels, bins=255, range=(1, 255))
        prob = hist / area
        prob_nz = prob[prob > 0]
        raw_ent = -np.sum(prob_nz * np.log2(prob_nz))
        return raw_sat, raw_skw, raw_ent
    # Extract raw data from both images
    sat1, skw1, ent1 = engine(cloud1)
    sat2, skw2, ent2 = engine(cloud2)
    # Normalization functions
    def getSat(val): 
        # Maps 0-255 to 0-100
        return (val / 255.0) * 100
    def getSkw(val):
        # Inverse mapping: Negative Skew (-2) -> 100, Positive (+2) -> 1
        val_clamp = max(-2.0, min(2.0, val))
        norm_val = 100 - ((val_clamp + 2) / 4) * 99
        return max(1.0, min(100.0, norm_val))
    def getEnt(val):
        # Maps entropy 0-8 to 1-100
        norm_val = 1 + (val / 8.0) * 99
        return max(1.0, min(100.0, norm_val))
    # Calculate scores for each metric averaged across the two images
    score = [
        round(np.mean([getSat(sat1), getSat(sat2)]), 1),
        round(np.mean([getSkw(skw1), getSkw(skw2)]), 1),
        round(np.mean([getEnt(ent1), getEnt(ent2)]), 1)
    ]
    return score
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
        # try:
        #     main_root = Path(Config.Paths.mainRooot)
        #     objRoot = (
        #         main_root /
        #         Config.Paths.DataRoots.ResourcesRoot /
        #         Config.Paths.DataRoots.StreamRoot / 
        #         Config.Paths.DataRoots.CaseStudyRoot() /
        #         Config.Packages.Drivers.__name__ / 
        #         Config.Packages.Drivers.Phases.Phase3.__name__ / 
        #         Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
        #         Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__ )
        #     maskRoot = (
        #         main_root /
        #         Config.Paths.DataRoots.ResourcesRoot /
        #         Config.Paths.DataRoots.StreamRoot / 
        #         Config.Paths.DataRoots.CaseStudyRoot() /
        #         Config.Packages.Drivers.__name__ / 
        #         Config.Packages.Drivers.Phases.Phase3.__name__ / 
        #         Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ /
        #         Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2.__name__ )
        #     dstRoot = (
        #         main_root /
        #         Config.Paths.DataRoots.ResourcesRoot /
        #         Config.Paths.DataRoots.StreamRoot / 
        #         Config.Paths.DataRoots.CaseStudyRoot() /
        #         Config.Packages.Drivers.__name__ / 
        #         Config.Packages.Drivers.Phases.Phase3.__name__ / 
        #         Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
        #         task_conf.__name__ /
        #         task_conf.MetaData.OutputExt)
        #     cavityData = pd.read_pickle((objRoot / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.CavityExt))
        #     cloudData = pd.read_pickle((objRoot / Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.CloudExt))
        #     ppr = Config.Settings.Acquisition.PPR 
        #     blades = Config.Settings.Acquisition.Blades
        #     files = [f.name for f in maskRoot.iterdir()]
        #     rows = []
        #     for idx, file in enumerate(files):
        #         key = Path(file).stem.split('_')[1]
        #         row = {
        #             'key': key,
        #             'Phase': int(key) % ppr,
        #             'Blade': ((int(key) % ppr) * blades // ppr) + 1}
        #         try:
        #             masks = pd.read_pickle((maskRoot / file))
        #             cavity1 = np.maximum.reduce(
        #                 [cell[0] for cell in masks.iloc[:, 0]])
        #             cavity2 = np.maximum.reduce(
        #                 [cell[0] for cell in masks.iloc[:, 1]])
        #             cloud1 = np.maximum.reduce(
        #                 [cell[1] for cell in masks.iloc[:, 0]])
        #             cloud2 = np.maximum.reduce(
        #                 [cell[1] for cell in masks.iloc[:, 1]])
        #             cavity, cloud = cavityData[key], cloudData[key]
        #         except Exception as e:
        #             continue
        #         for col in ['Area', 'Volume']:
        #             row[col] = [cavity[col].sum(), cloud[col].sum()]
        #         for col in ['Inner', 'Outer', 'Control']:
        #             cavity_vals = cavity[col].dropna().values
        #             cloud_vals = cloud[col].dropna().values
        #             row[col] = [
        #                 np.concatenate(cavity_vals) if len(cavity_vals) > 0 else np.array([]),
        #                 np.concatenate(cloud_vals) if len(cloud_vals) > 0 else np.array([])]
        #         row['Metrix'] = [getMetrix(cavity1, cavity2), getMetrix(cloud1, cloud2)]
        #         rows.append(row)
        #     aggregate = pd.DataFrame(rows) 
        #     pickle.dump(aggregate, open(dstRoot, 'wb'))
        #     print('.... Task2:', colored('Executed ✅', 'green'))
        # except Exception as e:
        #     print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
        #     raise e
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return