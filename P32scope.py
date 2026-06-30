key = '00001'

from Config import Config
from pathlib import Path
import pandas as pd
import numpy as np
import pyvista as pv
main_root = Path(Config.Paths.mainRooot)
root = (
    main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ / 
    Config.Packages.Drivers.Phases.Phase3.__name__ / 
    Config.Packages.Drivers.Phases.Phase3.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.__name__ /
    Config.Packages.Drivers.Phases.Phase3.Modules.Module2.Tasks.Task1.MetaData.OutputExt
)
data = pd.read_pickle(root)
if key not in data:
    print(f'Chosen Key is not present. Available keys:\n{list(data.keys())}')
else:
    pose_data = data[key]
    res = {}
    for col in ['Cavity', 'Cloud']:
        inner_list, mesh_list, total_area, total_vol = [], [], 0.0, 0.0
        for entry in pose_data[col]:
            inner_list.append(entry['Inner'])
            mesh_list.append(entry['Mesh'])
            total_area += entry['Area']
            total_vol += entry['Volume']
        res[f"{col}Inner"] = np.concatenate(inner_list, axis=0)
        res[f"{col}Mesh"] = np.concatenate(mesh_list, axis=0)
        res[f"{col}Area"] = total_area
        res[f"{col}Volume"] = total_vol
    base_points = pose_data.get('Pose', np.empty((0, 3)))
    if isinstance(base_points, pd.Series):
        if len(base_points) == 1:
            base_points = base_points.iloc[0]
        else:
            base_points = np.vstack(base_points.tolist())
    base_geom = pv.PolyData(np.asarray(base_points))
    pl = pv.Plotter(shape=(1, 2))
    pl.add_text(f"Key: {key}", font_size=14, position='upper_edge')
    vs = 2.0  
    voxel_cube = pv.Cube(x_length=vs, y_length=vs, z_length=vs)

    # PLOT 1: CAVITY
    pl.subplot(0, 0)
    cavity_info = f"Cavity\nArea: {res['CavityArea']:.2f}\nVolume: {res['CavityVolume']:.2f}"
    pl.add_text(cavity_info, font_size=11)
    pl.add_mesh(base_geom, color='darkorange', style='points', point_size=1, render_points_as_spheres=True)
    if len(res['CavityMesh']) > 0:
        pc_cav_mesh = pv.PolyData(res['CavityMesh'])
        cavity_voxels = pc_cav_mesh.glyph(geom=voxel_cube, orient=False, scale=False)
        pl.add_mesh(cavity_voxels, color='cyan', opacity=1, show_edges=True)
    pl.add_mesh(pv.PolyData(res['CavityInner']), color='royalblue', style='points', point_size=2, render_points_as_spheres=True)
    # PLOT 2: CLOUD
    pl.subplot(0, 1)
    cloud_info = f"Cloud\nArea: {res['CloudArea']:.2f}\nVolume: {res['CloudVolume']:.2f}"
    pl.add_text(cloud_info, font_size=11)
    pl.add_mesh(base_geom, color='darkorange', style='points', point_size=1, render_points_as_spheres=True)
    if len(res['CloudMesh']) > 0:
        pc_cloud_mesh = pv.PolyData(res['CloudMesh'])
        cloud_voxels = pc_cloud_mesh.glyph(geom=voxel_cube, orient=False, scale=False)
        pl.add_mesh(cloud_voxels, color='cyan', opacity=1, show_edges=True)
    pl.add_mesh(pv.PolyData(res['CloudInner']), color='royalblue', style='points', point_size=2, render_points_as_spheres=True)
    pl.link_views()
    pl.show(jupyter_backend='trame')