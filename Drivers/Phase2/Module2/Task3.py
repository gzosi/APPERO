#%% Importing Libraries
from pathlib import Path
import pandas as pd
import copy
import cv2 as cv
import numpy as np
import open3d as o3d
from tqdm import tqdm
import pickle
from termcolor import colored
#%% Defining Subroutines
def triangulationEngine(pts1, pts2, calib):
    ''' Triangola punti 2D stereo in coordinate 3D spaziali rimuovendo la distorsione radiale e tangenziale. '''
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)
    if pts1.size == 0 or pts2.size == 0:
        return np.empty((0, 3))
    udist1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), calib['K1'], calib['D1'], P = calib['K1']) 
    udist2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), calib['K2'], calib['D2'], P = calib['K2'])
    P1 = calib['K1'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = calib['K2'] @ np.hstack((calib['R'], calib['T']))
    points_4d_h = cv.triangulatePoints(
        P1, P2, udist1.reshape(-1, 2).T, udist2.reshape(-1, 2).T)
    return (points_4d_h[:3] / points_4d_h[3]).T
def localFilter(pcd, filter_settings):
    ''' Filtro leggero in fase di accumulo '''
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=filter_settings.Stats.nbNeighbors, 
        std_ratio=filter_settings.Stats.stdRatio)
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=filter_settings.Radius.nbPoints, 
        radius=filter_settings.Radius.radius)
    return pcd
def globalFilter(pcd, s):
    ''' 
    Esegue tutto il preprocessing: Taglio posizionale, Voxel, SOR, ROR e PCA.
    Restituisce la point cloud pronta per il meshing.
    '''
    if hasattr(s, 'Positional') and getattr(s.Positional, 'enabled', False):
        points = np.asarray(pcd.points)
        keep_mask = np.ones(len(points), dtype=bool)
        for zone in s.Positional.exclusion_zones:
            min_bound = np.array(zone[0])
            max_bound = np.array(zone[1])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            indices_inside = bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
            keep_mask[indices_inside] = False
        pcd = pcd.select_by_index(np.where(keep_mask)[0])
    if hasattr(s, 'Size') and s.Size.voxelSize > 0:
        pcd = pcd.voxel_down_sample(voxel_size=s.Size.voxelSize)
    if hasattr(s, 'Stats') and getattr(s.Stats, 'enabled', False):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=s.Stats.nbNeighbors, std_ratio=s.Stats.stdRatio)
    if hasattr(s, 'Radius') and getattr(s.Radius, 'enabled', False):
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=s.Radius.nbPoints, radius=s.Radius.radius)
    if hasattr(s, 'PCA') and getattr(s.PCA, 'enabled', False):
        pcd.estimate_covariances(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=s.PCA.search_radius, max_nn=s.PCA.max_nn))
        covariances = np.asarray(pcd.covariances)
        eigenvalues, _ = np.linalg.eigh(covariances)
        sum_eigenvalues = np.sum(eigenvalues, axis=1) + 1e-6
        surface_variation = eigenvalues[:, 0] / sum_eigenvalues
        mask = surface_variation < s.PCA.threshold
        pcd = pcd.select_by_index(np.where(mask)[0])
    if len(pcd.points) > 0:
        eps_val = s.Size.voxelSize * 5.0 if (hasattr(s, 'Size') and s.Size.voxelSize > 0) else 10.0
        labels = np.array(pcd.cluster_dbscan(eps=eps_val, min_points=20, print_progress=False))
        if len(labels) > 0:
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid_mask = unique_labels >= 0
            if np.any(valid_mask):
                largest_cluster_label = unique_labels[valid_mask][np.argmax(counts[valid_mask])]
                pcd = pcd.select_by_index(np.where(labels == largest_cluster_label)[0])
        if len(pcd.points) == 0:
            print(colored("ERRORE: Nuvola vuota dopo i filtri (Controlla la soglia PCA).", "red")) 
    return pcd
def meshEngine(pcd, s):
    ''' 
    Riceve una point cloud GIA' FILTRATA. 
    Calcola normali, genera la mesh (Poisson), la pulisce, rimuove i frammenti sparsi, 
    la liscia ed estrae i dati di output.
    '''
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=s.NormalOrient.radius, max_nn=s.NormalOrient.max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=s.NormalOrient.max_nn)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=s.Poisson.depth, width=s.Poisson.width, 
        scale=s.Poisson.scale, linear_fit=s.Poisson.linear_fit)
    if hasattr(s.Poisson, 'density_trim') and s.Poisson.density_trim > 0:
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, s.Poisson.density_trim * 0.1)
        mesh.remove_vertices_by_mask(densities < density_threshold)
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
    if hasattr(s.Cleaning, 'max_distance_from_source') and s.Cleaning.max_distance_from_source > 0:
        mesh_vertices_pcd = o3d.geometry.PointCloud()
        mesh_vertices_pcd.points = mesh.vertices
        dists = np.asarray(mesh_vertices_pcd.compute_point_cloud_distance(pcd))
        mesh.remove_vertices_by_mask(dists > s.Cleaning.max_distance_from_source)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = np.asarray(cluster_n_triangles).argmax()
        triangles_to_remove = np.asarray(triangle_clusters) != largest_cluster_idx
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
    if len(np.asarray(mesh.triangles)) == 0:
        print(colored("ERRORE: Mesh vuota dopo il filtering.", "red"))
        return mesh, np.array([]), np.array([])
    if hasattr(s.Smoothing, 'subdivide') and s.Smoothing.subdivide > 0:
        mesh = mesh.subdivide_midpoint(number_of_iterations=s.Smoothing.subdivide)
    if hasattr(s.Smoothing, 'method') and s.Smoothing.method == 'laplacian':
        mesh = mesh.filter_smooth_simple(number_of_iterations=s.Smoothing.iterations)
    elif hasattr(s.Smoothing, 'method') and s.Smoothing.method == 'taubin':
        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=s.Smoothing.iterations,
            lambda_filter=s.Smoothing.lambda_filter, mu=s.Smoothing.mu)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    centroids = np.mean(vertices[triangles], axis=1)
    v0, v1, v2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return mesh, centroids, areas
#%% Defining Main Function
def main(Config):    
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3
    if task_conf.General.Activation is True:
        print('.... Task3:', colored('Running ℹ️', 'cyan'))
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
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task2.MetaData.OutputExt)
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
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.OutputExt)
        meshRoot = (
            main_root / 
            Config.Paths.DataRoots.ResourcesRoot / 
            Config.Paths.DataRoots.StreamRoot / 
            Config.Paths.DataRoots.CaseStudyRoot() /
            Config.Packages.Drivers.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__ / 
            Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.MeshExt)
        settings = task_conf.Settings
        calib = pd.read_pickle(calibRoot)[
            settings.Calib.Dataset][settings.Calib.Pair][settings.Calib.Model]
        poses = pd.read_pickle(poseRoot)['BackwardSync']
        clouds = {
            phase: localFilter(
                (pcd := o3d.geometry.PointCloud(),
                setattr(pcd, 'points', o3d.utility.Vector3dVector(np.concatenate([
                triangulationEngine(np.array(row['Camera1']), np.array(row['Camera2']), calib)
                for _, row in group.iterrows()]))), pcd)[2],
                settings.LocalFilter)
            for phase, group in pd.read_pickle(stereoRoot).groupby('Phase')}
        try:
            all_points = []
            for phase, cloud in clouds.items():
                if not any(phase in np.arange(a, b) for a, b in settings.Bounds): 
                    continue
                temp_cloud = copy.deepcopy(cloud) 
                temp_cloud.transform(poses[phase])
                all_points.append(temp_cloud)
            sync = o3d.geometry.PointCloud()
            for p in all_points:
                sync += p
            sync = sync.voxel_down_sample(voxel_size=settings.LocalFilter.Size.voxelSize)
            globalPcd = globalFilter(sync, settings.GlobalFilter)
            mesh, centroids, areas = meshEngine(globalPcd, settings.Mesh)
            o3d.io.write_triangle_mesh(meshRoot, mesh)
            pickle.dump(dict(areas = areas, pts = centroids), open(dstRoot, 'wb'))
            print('.... Task3:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task3:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task3:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task3 Switch (on/off) ❌')
    return 