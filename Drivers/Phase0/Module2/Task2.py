#%% Importing Libreries
import h5py
import json
from pathlib import Path
from termcolor import colored
#%% Defining Subroutines
def exploreFile(group, settings):
    ''' Esplora ricorsivamente il file HDF5 e ne estrae dimensioni e origini. '''
    structure_shapes = {}
    structure_origins = {}
    full_w, full_h = settings.FullSensorShape
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            sub_shapes, sub_origins = exploreFile(item, settings)
            structure_shapes[key] = sub_shapes
            structure_origins[key] = sub_origins
        elif isinstance(item, h5py.Dataset):
            current_shape = item.shape
            shape_val = [int(x) for x in current_shape] 
            img_h = int(current_shape[0])
            img_w = int(current_shape[1])
            if settings.IncludeOrigin:
                origin_x = int((full_w - img_w) / 2) if img_w < full_w else 0
                origin_y = int((full_h - img_h) / 2) if img_h < full_h else 0
                origin_val = [origin_x, origin_y]
            else:
                origin_val = None
            return shape_val, origin_val
    return structure_shapes, structure_origins
#%% Defining Main Function
def main(Config):
    task_conf = Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task2
    task_settings = task_conf.Settings
    task_general = task_conf.General
    task_meta = task_conf.MetaData
    if task_general.Activation is True:
        print('.... Task2:', colored( 'Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        baseFolder = (main_root /
                      Config.Paths.DataRoots.ResourcesRoot /
                      Config.Paths.DataRoots.StreamRoot /
                      Config.Paths.DataRoots.CaseStudyRoot() /
                      Config.Packages.Drivers.__name__ / 
                      Config.Packages.Drivers.Phases.Phase0.__name__ /
                      Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
                      task_conf.__name__)
        if not baseFolder.exists():
            raise FileNotFoundError(f"Cartella di destinazione non trovata!")
        srcRoot = (main_root /
                   Config.Paths.DataRoots.ResourcesRoot /
                   Config.Paths.DataRoots.StreamRoot /
                   Config.Paths.DataRoots.CaseStudyRoot() /
                   Config.Packages.Drivers.__name__ / 
                   Config.Packages.Drivers.Phases.Phase0.__name__ /
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)
        dstShapePath = baseFolder / task_meta.ShapeExt
        dstOriginPath = baseFolder / task_meta.OriginExt
        indent = task_meta.Indent
        try:
            with h5py.File(srcRoot, 'r') as f:
                shapes_data, origins_data = exploreFile(f, task_settings)
            with open(dstShapePath, 'w') as f:
                json.dump(shapes_data, f, indent=indent)
            with open(dstOriginPath, 'w') as f:
                json.dump(origins_data, f, indent=indent)
            print('.... Task2:', colored( 'Executed ✅', 'green'))
        except Exception as e:
            print('.... Task2:', colored( f'Error: {e} ❌', 'red'))
            raise e
    elif task_general.Activation is False:
        print('.... Task2:', colored( 'Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task2 Switch (on/off) ❌')
    return