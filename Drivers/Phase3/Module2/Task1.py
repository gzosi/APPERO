#%% Importing Libreries
import cv2 as cv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.interpolate import RBFInterpolator
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_closing
import open3d as o3d
from tqdm import tqdm
from termcolor import colored
import pyvista as pv
#%% Defining Subroutines
def triangulationEngine(pts1: np.ndarray, pts2: np.ndarray, calib: dict) -> np.ndarray:
    """Triangola punti 2D stereo in coordinate 3D spaziali rimuovendo la distorsione."""
    if len(pts1) == 0 or len(pts2) == 0:
        return np.empty((0, 3))
    pts1_f32 = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    pts2_f32 = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
    udist1 = cv.undistortPoints(pts1_f32, calib['K1'], calib['D1'], P=calib['K1']) 
    udist2 = cv.undistortPoints(pts2_f32, calib['K2'], calib['D2'], P=calib['K2'])
    P1 = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = calib['K2'] @ np.hstack((calib['R'], calib['T']))
    points_4d_h = cv.triangulatePoints(P1, P2, udist1.reshape(-1, 2).T, udist2.reshape(-1, 2).T)
    return (points_4d_h[:3] / points_4d_h[3]).T
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
def islandSplit(pts: np.ndarray, ids: np.ndarray, config) -> tuple:
    """Suddivide i punti intagliati in isole contigue e calcola le normali orientate."""
    ids = np.asarray(ids)
    if len(pts) == 0: return ([], [], []) 
    tree = cKDTree(pts)
    pairs = tree.query_pairs(config.Island.MaxDist)
    if not pairs: return ([pts], [ids], [np.zeros((len(pts), 3))]) 
    pairs_arr = np.array(list(pairs))
    adj_matrix = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs_arr[:, 0], pairs_arr[:, 1])), shape=(len(pts), len(pts)))
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    islands, islandIds, normals = [], [], []
    pcd = o3d.geometry.PointCloud()
    for i in range(n_components):
        mask = (labels == i)
        island = pts[mask]
        if len(island) > 3: 
            pcd.points = o3d.utility.Vector3dVector(island)
            pcd_tiles = pcd.voxel_down_sample(config.Normals.TileSize)
            if len(pcd_tiles.points) < 3: pcd_tiles = pcd
            pcd_tiles.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config.Normals.TileSize*2.5, max_nn=config.Normals.Max_nn))
            pcd_tiles.orient_normals_towards_camera_location(np.array(config.Normals.CameraPose))
            _, tile_indices = cKDTree(np.asarray(pcd_tiles.points)).query(island, k=1) 
            clean_normals = np.asarray(pcd_tiles.normals)[tile_indices]
            islands.append(island)
            islandIds.append(ids[mask])
            normals.append(clean_normals.copy())
    return (islands, islandIds, normals)
def getBoundary(pts: np.ndarray, config) -> tuple:
    """Estrae i bordi 3D calcolando l'Angle Gap sul piano tangente locale."""
    if len(pts) < config.k_neighbors: return list(range(len(pts))), pts
    distances, indices = cKDTree(pts).query(pts, k=config.k_neighbors)
    boundary_indices = []
    for i, pt in enumerate(pts):
        neighbor_pts = pts[indices[i]] 
        centered = neighbor_pts - np.mean(neighbor_pts, axis=0)
        _, eigenvectors = np.linalg.eigh(np.dot(centered.T, centered))
        normal = eigenvectors[:, 0] 
        u = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        neighbors_vec = neighbor_pts[1:] - pt
        angles = np.sort(np.arctan2(np.dot(neighbors_vec, v), np.dot(neighbors_vec, u)))
        if len(angles) < 2:
            boundary_indices.append(i)
            continue
        if max(np.max(np.diff(angles)), (angles[0] + 2 * np.pi) - angles[-1]) > np.radians(config.min_angle_deg):
            boundary_indices.append(i)
    if not boundary_indices: return [], np.empty((0, 3))
    bnd_pts = pts[boundary_indices]
    counts = np.array([len(nl) for nl in cKDTree(bnd_pts).query_ball_point(bnd_pts, r=np.mean(distances[:, 1]) * 4.0)])
    valid_mask = counts >= 3
    valid_idx = np.array(boundary_indices)[valid_mask].tolist()
    return valid_idx, pts[valid_idx]
def classBoundary(island_pts: np.ndarray, pose_bnd_pts: np.ndarray, config) -> tuple:
    """Divide i bordi in Veri e Falsi."""
    shell_bnd_idx, shell_bnd_pts = getBoundary(island_pts, config)
    if len(shell_bnd_pts) == 0 or len(pose_bnd_pts) == 0: return shell_bnd_idx, []
    distances, _ = cKDTree(pose_bnd_pts).query(shell_bnd_pts)
    mask = distances > config.Threshold
    return shell_bnd_pts[mask], shell_bnd_pts[~mask]
def getControls(innerPts: np.ndarray, innerNorm: np.ndarray, controlPts: np.ndarray, bnd: np.ndarray, config) -> np.ndarray:
    """Corregge i punti di controllo preservandone la forma (Rigid-like Local Shift)."""
    innerTree = cKDTree(innerPts)
    _, indices = innerTree.query(controlPts)
    raw_penetrations = np.maximum(0, -np.sum((controlPts - innerPts[indices]) * innerNorm[indices], axis=1))
    is_anchor = np.zeros(len(controlPts), dtype=bool)
    if len(bnd) > 0:
        dist_to_bnd, _ = cKDTree(bnd).query(controlPts)
        is_anchor = dist_to_bnd < (np.linalg.norm(np.max(controlPts, axis=0) - np.min(controlPts, axis=0)) * 0.01)
    _, neighbors = cKDTree(controlPts).query(controlPts, k=config.k_neighbors)
    D = raw_penetrations.copy()
    for _ in range(config.dilation_iters): D = np.max(D[neighbors], axis=1)
    for _ in range(config.smooth_iters):
        D = np.maximum(np.mean(D[neighbors], axis=1), raw_penetrations)
        D[is_anchor] = 0                       
    return controlPts + (innerNorm[indices] * D[:, np.newaxis])
def generateVolume(innerPts: np.ndarray, innerNorm: np.ndarray, bndPts: np.ndarray, controlPts: np.ndarray, config) -> tuple:
    """
    Genera l'outerShell e la mesh a voxel (volume).
    """
    if len(innerPts) == 0: return np.empty((0, 3)), np.empty((0, 3)), 0.0, np.array([])
    vs = config.voxelSize
    innerTree = cKDTree(innerPts)
    _, indices = innerTree.query(controlPts)
    thicknesses = np.sum((controlPts - innerPts[indices]) * innerNorm[indices], axis=1)
    valid_mask = thicknesses > 0
    known_pts_list, known_thick_list = list(innerPts[indices][valid_mask]), list(thicknesses[valid_mask])
    if len(bndPts) > 0:
        for idx in np.unique(innerTree.query(bndPts)[1]):
            known_pts_list.append(innerPts[idx])
            known_thick_list.append(0.0)  
    known_pts, known_thicknesses = np.array(known_pts_list), np.array(known_thick_list)    
    if len(known_pts) < 4:
        T_field = np.zeros(len(innerPts))
    else:
        try:
            rbf = RBFInterpolator(known_pts, known_thicknesses, kernel=config.rbfKernel, smoothing=config.rbfSmoothing)
            T_field = np.maximum(rbf(innerPts), 0.0)
        except Exception as e:
            T_field = np.full(len(innerPts), np.mean(known_thicknesses) if len(known_thicknesses) > 0 else 0.0)            
    outerPts = innerPts + innerNorm * T_field[:, np.newaxis]
    step_fraction = getattr(config, 'rayStepFraction', 3.0)
    step_size = vs / step_fraction
    max_T = np.max(T_field) if len(T_field) > 0 else 0.0
    if max_T > 0:
        num_steps = int(np.ceil(max_T / step_size)) + 1
        steps = np.arange(num_steps) * step_size
        valid_steps_mask = steps[np.newaxis, :] <= T_field[:, np.newaxis]
        dense_cloud = innerPts[:, np.newaxis, :] + innerNorm[:, np.newaxis, :] * steps[np.newaxis, :, np.newaxis]
        valid_points = dense_cloud[valid_steps_mask]
        base_indices_grid = np.arange(len(innerPts))[:, np.newaxis]
        base_indices_grid = np.broadcast_to(base_indices_grid, valid_steps_mask.shape)
        valid_base_indices = base_indices_grid[valid_steps_mask]
    else:
        valid_points = innerPts
        valid_base_indices = np.arange(len(innerPts))
    grid_coords = np.round(valid_points / vs).astype(int)    
    _, unique_idx = np.unique(grid_coords, axis=0, return_index=True)
    unique_coords = grid_coords[unique_idx]
    unique_base_indices = valid_base_indices[unique_idx] 
    ray_iters = getattr(config, 'rayClosingIters', 2)
    if ray_iters > 0 and len(unique_coords) > 0:
        min_c = np.min(unique_coords, axis=0)
        max_c = np.max(unique_coords, axis=0)
        pad = ray_iters + 2
        shape = max_c - min_c + (pad * 2) 
        grid_3d = np.zeros(shape, dtype=bool)
        offset_coords = unique_coords - min_c + pad
        grid_3d[offset_coords[:, 0], offset_coords[:, 1], offset_coords[:, 2]] = True
        grid_filled = binary_closing(grid_3d, iterations=ray_iters)
        new_offset_coords = np.argwhere(grid_filled)
        final_grid_coords = new_offset_coords + min_c - pad
        tree_orig = cKDTree(unique_coords)
        _, nearest_idx = tree_orig.query(final_grid_coords)
        final_base_indices = unique_base_indices[nearest_idx]
    else:
        final_grid_coords = unique_coords
        final_base_indices = unique_base_indices
    interiorVoxels = final_grid_coords * vs
    voxel_base_indices = final_base_indices 
    volumeValue = len(interiorVoxels) * (vs ** 3)
    return outerPts, interiorVoxels, volumeValue, voxel_base_indices
def process(obj_data, poses, phase, calib, areas, settings) -> dict:
    """Processa un singolo frame triangolando e intagliando il volume."""
    empty_cavity = {'Inner': np.empty((0, 3)), 'Mesh': np.empty((0, 3)), 'Area': 0.0, 'Volume': 0.0}
    empty_cloud = {'Inner': np.empty((0, 3)), 'Mesh': np.empty((0, 3)), 'Area': 0.0, 'Volume': 0.0}
    results = {'Pose': np.empty((0, 3)), 'Cavity': empty_cavity, 'Cloud': empty_cloud}
    cavity1, cloud1, cavPts1, _ = obj_data[0]
    cavity2, cloud2, cavPts2, _ = obj_data[1]
    try:
        cavPts = triangulationEngine(cavPts1, cavPts2, calib)
        pose = poses[phase]
        if pose is None or len(pose) == 0:
            return results
        results['Pose'] = pose
    except Exception as e:
        print(colored(f"    [!] Errore Triangolazione/Posa: {e}", 'yellow'))
        return results
    valid_ids_cav = np.empty(0, dtype=int)
    voxel_base_indices = np.empty(0, dtype=int)
    voxels_cav = np.empty((0, 3))
    try:
        _, bndsP = getBoundary(pose, settings.OuterShell.Boundary)
        id_cav, innerShell_cav = surfaceCarver([cavity1, cavity2], calib, pose, settings.InnerShell)
        area_cav = sum(areas[id_cav]) if id_cav.size > 0 else 0.0
        islands_cav, ids_cav, norms_cav = islandSplit(innerShell_cav, id_cav, settings.InnerShell) 
        bnds_cav = [classBoundary(island, bndsP, settings.OuterShell.Boundary)[0] for island in islands_cav]
        valid_inner_cav = np.concatenate(islands_cav) if islands_cav else np.empty((0, 3))
        valid_ids_cav = np.concatenate(ids_cav) if ids_cav else np.empty(0, dtype=int)
        normals_cav = np.concatenate(norms_cav) if norms_cav else np.empty((0, 3))
        boundaries_cav = np.concatenate(bnds_cav) if bnds_cav else np.empty((0, 3))
        controlShell_cav = np.concatenate([
            getControls(shell, norms_cav[i], cavPts, bnds_cav[i], settings.ControlShell) 
            for i, shell in enumerate(islands_cav)
        ]) if islands_cav else np.empty((0, 3))
        outerShell_cav, voxels_cav, vol_cav, voxel_base_indices = generateVolume(
            valid_inner_cav, normals_cav, boundaries_cav, controlShell_cav, settings.OuterShell.Volume)
        results['Cavity'] = {
            'Inner': valid_inner_cav, 'Mesh': voxels_cav, 'Area': area_cav, 'Volume': vol_cav}
    except Exception as e:
        print(colored(f"    [!] Errore Calcolo Cavity: {e}", 'yellow'))
    try:
        id_cloud, innerShell_cloud = surfaceCarver([cloud1, cloud2], calib, pose, settings.InnerShell)
        area_cloud = sum(areas[id_cloud]) if id_cloud.size > 0 else 0.0

        if len(voxels_cav) > 0 and len(id_cloud) > 0 and len(valid_ids_cav) > 0:
            pose_index_of_voxel = valid_ids_cav[voxel_base_indices]
            is_cloud_voxel = np.isin(pose_index_of_voxel, id_cloud)
            voxels_cloud = voxels_cav[is_cloud_voxel]
            vol_cloud = len(voxels_cloud) * (settings.OuterShell.Volume.voxelSize ** 3)
        else:
            voxels_cloud = np.empty((0, 3))
            vol_cloud = 0.0
        results['Cloud'] = {
            'Inner': innerShell_cloud, 'Mesh': voxels_cloud, 'Area': area_cloud, 'Volume': vol_cloud}
    except Exception as e:
        print(colored(f"    [!] Errore Calcolo Cloud: {e}", 'yellow'))
    return results
#%% Defining Main Function
def main(Config):       
    task_conf = Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1
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
            Config.Packages.Drivers.Phases.Phase3.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.Modules.Module1.Tasks.Task2.__name__ )
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
            Config.Packages.Drivers.Phases.Phase3.__name__ / 
            Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__ /
            Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.OutputExt)
        settings = task_conf.Settings
        period = Config.Settings.Acquisition.PPR
        calib = pd.read_pickle(calibRoot)[settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
        poses, areas = [pd.read_pickle(poseRoot)[k] for k in ['single', 'areas']]
        files = sorted(f.name for f in objRoot.iterdir() if f.is_file())
        allData = {}
        try:
            for name in tqdm(files, desc=colored('Surface Carving 🚀', 'magenta'), ncols=100): 
                key = ''.join(c for c in name if c.isdigit())
                phase = int(key) % period 
                if key not in allData:
                    allData[key] = pd.DataFrame(columns=['Pose', 'Cavity', 'Cloud'])
                objs = pd.read_pickle((objRoot / name))
                for i, (_, obj) in enumerate(objs.iterrows()):
                    try:
                        res = process(obj, poses, phase, calib, areas, settings)
                        allData[key].loc[i] = {
                            'Pose': res['Pose'],
                            'Cavity': res['Cavity'],
                            'Cloud': res['Cloud']}
                    except Exception as e:
                        print(colored(f"\n[FATAL] Errore Frame (Key: {key}, Iter: {i}): {str(e)}", 'red'))
                        allData[key].loc[i] = {
                            'Pose': np.empty((0,3)), 
                            'Cavity': {'Inner': np.empty((0, 3)), 'Mesh': np.empty((0, 3)), 'Area': 0.0, 'Volume': 0.0},
                            'Cloud': {'Inner': np.empty((0, 3)), 'Mesh': np.empty((0, 3)), 'Area': 0.0, 'Volume': 0.0}}
            pd.to_pickle(allData, dstRoot)
            print('.... Task1:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task1:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return