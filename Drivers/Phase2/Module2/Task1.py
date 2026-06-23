#%% Importing Libraries
from pathlib import Path
import pandas as pd
import numpy as np
import cv2 as cv
from termcolor import colored
import open3d as o3d
from tqdm import tqdm
#%% Defining Subroutines
def triangulationEngine(pts1, pts2, calib):
    ''' Triangola punti 2D stereo in coordinate 3D spaziali rimuovendo la distorsione radiale e tangenziale. '''
    udist1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), calib['K1'], calib['D1'], P = calib['K1']) 
    udist2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), calib['K2'], calib['D2'], P = calib['K2'])
    P1 = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = calib['K2'] @ np.hstack((calib['R'], calib['T']))
    points_4d_h = cv.triangulatePoints(
        P1, P2, udist1.reshape(-1, 2).T, udist2.reshape(-1, 2).T)
    return (points_4d_h[:3] / points_4d_h[3]).T
def kabsch(A, B):
    ''' 
    Algoritmo di Kabsch: calcola R e t tali che B = R*A + t.
    Usa SVD per minimizzare l'errore quadratico medio (RMS).
    A e B devono avere shape (N, 3)
    '''
    assert A.shape == B.shape
    num_pts = A.shape[0]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
def icp(source_pts, target_pts, init_transform, threshold):
    ''' Raffina la trasformazione grezza (Kabsch) usando l'algoritmo ICP di Open3D. '''
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pts)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pts)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation
#%% Defining Main Function
def main(Config):    
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task1
    if task_conf.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        stereoRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task1.MetaData.OutputExt)
        linksRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module1.Tasks.Task2.MetaData.OutputExt)
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
            Config.Packages.Drivers.__name__ / Config.Packages.Drivers.Phases.Phase2.__name__ /
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ /
            task_conf.__name__ /
            task_conf.MetaData.OutputExt)
        ppr = Config.Settings.Acquisition.PPR
        settings = task_conf.Settings
        try:
            stereo = pd.read_pickle(stereoRoot)
            links = pd.read_pickle(linksRoot)
            calib = pd.read_pickle(calibRoot)[
                settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
            icp_active = getattr(settings.ICP, 'Activation', False) if hasattr(settings, 'ICP') else False
            icp_thresh = getattr(settings.ICP, 'Threshold', 10.0) if hasattr(settings, 'ICP') else 10.0
            cam1, cam2 = list(stereo.columns[:2])
            data = []
            for (key, nextKey), row in tqdm(links.iterrows(), total=len(links), desc=colored(f'Pose Estimation 🚀', 'magenta'), ncols=100):
                curr_idxs, next_idxs = row['Current'], row['Next']
                anchors_curr1 = np.array(stereo.loc[key, cam1])[curr_idxs]
                anchors_curr2 = np.array(stereo.loc[key, cam2])[curr_idxs]
                anchors_next1 = np.array(stereo.loc[nextKey, cam1])[next_idxs]
                anchors_next2 = np.array(stereo.loc[nextKey, cam2])[next_idxs]
                anchors3D_curr = triangulationEngine(anchors_curr1, anchors_curr2, calib)
                anchors3D_next = triangulationEngine(anchors_next1, anchors_next2, calib)
                T_Forward_Init = kabsch(anchors3D_curr, anchors3D_next)
                T_Backward_Init = kabsch(anchors3D_next, anchors3D_curr)
                if icp_active:
                    full_curr1 = np.array(stereo.loc[key, cam1])
                    full_curr2 = np.array(stereo.loc[key, cam2])
                    full_next1 = np.array(stereo.loc[nextKey, cam1])
                    full_next2 = np.array(stereo.loc[nextKey, cam2])
                    full3D_curr = triangulationEngine(full_curr1, full_curr2, calib)
                    full3D_next = triangulationEngine(full_next1, full_next2, calib)
                    T_Forward = icp(full3D_curr, full3D_next, T_Forward_Init, icp_thresh)
                    T_Backward = icp(full3D_next, full3D_curr, T_Backward_Init, icp_thresh)
                else:
                    T_Forward = T_Forward_Init
                    T_Backward = T_Backward_Init    
                data.append({
                    'Keys': (key, nextKey),
                    'Phase': (int(key)%ppr, int(nextKey)%ppr),
                    'Forward': T_Forward,
                    'Backward': T_Backward})
            if len(data) > 0:
                final_data = pd.DataFrame(data).set_index('Keys')
                final_data.to_pickle(dstRoot)
            else: 
                raise ValueError('Nessuna trasformazione calcolata.')
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return 