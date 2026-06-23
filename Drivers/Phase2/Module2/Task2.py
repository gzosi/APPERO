#%% Importing Libraries
from pathlib import Path
import copy
import cv2 as cv
import numpy as np
import pandas as pd
import open3d as o3d
import pyvista as pv
from scipy.spatial.transform import Rotation
from scipy.linalg import fractional_matrix_power
from tqdm import tqdm
import pickle
from termcolor import colored
#%% Defining Subroutines
def purifyMotion(T, ppr):
    '''
    Forza la matrice ad essere una rotazione perfetta attorno al suo asse.
    Corregge l'angolo per chiudere esattamente 360° in 'ppr' step,
    e azzera la traslazione assiale per impedire l'effetto "vite" (drift).
    '''
    T_pure = np.eye(4)
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-6:
        return T.copy()
    axis = rotvec / angle
    d = np.dot(axis, t)
    t_perp = t - d * axis
    I = np.eye(3)
    c = np.linalg.pinv(I - R) @ t_perp
    ideal_angle = (2.0 * np.pi) / ppr
    ideal_rotvec = axis * ideal_angle
    ideal_R = Rotation.from_rotvec(ideal_rotvec).as_matrix()
    ideal_t = (I - ideal_R) @ c
    T_pure[0:3, 0:3] = ideal_R
    T_pure[0:3, 3] = ideal_t
    return T_pure
def power_transform(T, power):
    ''' 
    Calcola la potenza usando fractional_matrix_power come richiesto, ma 
    estrae esplicitamente la parte reale e ri-ortogonalizza la matrice di 
    rotazione per evitare derive (drift) e ComplexWarning.
    '''
    T_pow_complex = fractional_matrix_power(T, power)
    T_pow = np.real(T_pow_complex).copy()
    U, _, Vt = np.linalg.svd(T_pow[0:3, 0:3])
    T_pow[0:3, 0:3] = U @ Vt
    return T_pow
def triangulationEngine(pts1, pts2, calib):
    ''' Triangola punti 2D stereo in coordinate 3D spaziali rimuovendo la distorsione radiale e tangenziale. '''
    udist1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), calib['K1'], calib['D1'], P = calib['K1']) 
    udist2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), calib['K2'], calib['D2'], P = calib['K2'])
    P1 = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = calib['K2'] @ np.hstack((calib['R'], calib['T']))
    points_4d_h = cv.triangulatePoints(
        P1, P2, udist1.reshape(-1, 2).T, udist2.reshape(-1, 2).T)
    return (points_4d_h[:3] / points_4d_h[3]).T
def averageEngine(Ts):
    ''' Calcola la media di una lista di matrici di rototraslazione. '''
    n = len(Ts)
    if n == 0:
        raise ValueError("La lista delle matrici è vuota.")
    Rs = np.zeros((n, 3, 3))
    ts = np.zeros((n, 3))
    for i, T in enumerate(Ts):
        Rs[i] = T[0:3, 0:3]
        ts[i] = T[0:3, 3]
    avgT = np.eye(4)
    avgT[0:3, 0:3] = Rotation.from_matrix(Rs).mean().as_matrix()
    avgT[0:3, 3] = np.mean(ts, axis=0)
    return avgT
def localFilter(pcd, filter_settings):
    ''' Filtro leggero in fase di accumulo '''
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=filter_settings.Stats.nbNeighbors, 
        std_ratio=filter_settings.Stats.stdRatio)
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=filter_settings.Radius.nbPoints, 
        radius=filter_settings.Radius.radius)
    return pcd
def icp(source_data, target_pcd, config_icp, init_transform=np.eye(4)):
    ''' Registrazione ICP avanzata '''
    if isinstance(source_data, np.ndarray):
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_data)
    else:
        source_pcd = source_data  
    if config_icp.usePointToPlane:
        if not source_pcd.has_normals():
            source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
            source_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.])) 
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
            target_pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
        method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    registration = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, config_icp.maxDistance, init_transform,
        method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config_icp.maxIterations))
    return registration.transformation
#%% Defining Main Function
def main(Config):    
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2
    if task_conf.General.Activation is True:
        print('.... Task2:', colored('Running ℹ️', 'cyan'))
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
        poseRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task1.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
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
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.MetaData.OutputExt)
        settings = task_conf.Settings
        ppr = Config.Settings.Acquisition.PPR
        blades = Config.Settings.Acquisition.Blades
        try:
            stereo = pd.read_pickle(stereoRoot)
            pose = pd.read_pickle(poseRoot)
            calib = pd.read_pickle(calibRoot)[
                settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
            backwardT, forwardT = [
                averageEngine(pose[k]) for k in ('Backward', 'Forward')]
            period = int(ppr / blades) if ppr % blades == 0 else ppr
            clouds = {
                phase: localFilter(
                    (pcd := o3d.geometry.PointCloud(),
                    setattr(pcd, 'points', o3d.utility.Vector3dVector(np.concatenate([
                    triangulationEngine(np.array(row['Camera1']), np.array(row['Camera2']), calib)
                    for _, row in group.iterrows()]))), pcd)[2],
                    settings.LocalFilter)
                for phase, group in stereo.groupby(stereo['Phase'] % period)
            }
            phases, scroll = np.arange(period), np.arange(settings.Scroll) + 1
            precomputed_init_B = {s: power_transform(backwardT, s) for s in scroll}
            precomputed_init_F = {s: power_transform(forwardT, s) for s in scroll}
            Bs, Fs = [], []
            for phase in tqdm(phases, total=len(phases), desc=colored('Pose Refinement 🚀', 'magenta'), ncols=100):
                source = clouds[phase]
                for s in scroll:
                    shift = (phase + s) % period
                    target = clouds[shift]
                    init_B = precomputed_init_B[s]
                    init_F = precomputed_init_F[s]
                    B = icp(target, source, settings.ICP, init_B)
                    F = icp(source, target, settings.ICP, init_F)
                    Bs.append(power_transform(B, 1.0 / s))
                    Fs.append(power_transform(F, 1.0 / s))
            bestF = averageEngine(Fs)
            bestB = averageEngine(Bs)
            bestF_pure = purifyMotion(bestF, ppr)
            bestB_pure = purifyMotion(bestB, ppr)
            PropFs = {n: power_transform(bestF_pure, n) for n in range(ppr)}
            PureBs = {n: power_transform(bestB_pure, n) for n in range(ppr)}
            SyncFs = {n: power_transform(bestF_pure, n % (ppr / blades)) for n in range(ppr)}
            SyncBs = {n: power_transform(bestB_pure, n % (ppr / blades)) for n in range(ppr)}
            BlaseFs = {n: [
                    power_transform(bestF_pure, (n + j * (ppr / blades)) % ppr) 
                    for j in range(blades)]
                for n in range(ppr)}
            BlaseBs = {n: [
                    power_transform(bestB_pure, (n + j * (ppr / blades)) % ppr) 
                    for j in range(blades)]
                for n in range(ppr)}
            pickle.dump(dict(
                ForwardProp = PropFs,
                BackwardProp = PureBs, 
                ForwardSync = SyncFs,
                BackwardSync = SyncBs,
                ForwardBlade = BlaseFs,
                BackwardBlade = BlaseBs
            ), open(dstRoot, 'wb'))
            print('.... Task2:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task2:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task2:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return 