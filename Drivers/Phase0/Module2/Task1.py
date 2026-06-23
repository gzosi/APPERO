#%% Importing Libreries
import cv2 as cv
import h5py
import inspect
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from termcolor import colored
#%% Defining Subroutines
def flatten_resync_class(cls, current_path=""):
    ''' 
    Esplora ricorsivamente la classe Resync e genera un dizionario piatto.
    Esempio: {'Database3/Dataset2/Foreground': 24, 'Database2/Dataset1': 17}
    '''
    flat_dict = {}
    if not inspect.isclass(cls):
        return flat_dict
    for key, value in cls.__dict__.items():
        if key.startswith('__'):
            continue
        new_path = f"{current_path}/{key}" if current_path else key
        if isinstance(value, int):
            flat_dict[new_path] = value
        elif inspect.isclass(value):
            flat_dict.update(flatten_resync_class(value, new_path))
    return flat_dict
def directoryExplorer(Config, srcRoot_str, dstRoot_str):
    ''' Raggruppa le immagini, applica Resync e Crop, e salva in HDF5. '''
    srcRoot, dstRoot = Path(srcRoot_str), Path(dstRoot_str)
    if not dstRoot.exists():
        print(colored(f'CRITICAL ERROR ❌: La cartella di destinazione non esiste!\nPath: {dstRoot}', 'red'))
        return
    task_config = Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1
    inputExt = task_config.MetaData.InputExt
    outputName = task_config.MetaData.OutputName
    ppr = Config.Settings.Acquisition.PPR
    resync_class = getattr(task_config.Settings, 'Resync', None)
    resync_dict = flatten_resync_class(resync_class) if resync_class else {}
    grouped_files = defaultdict(list)
    for ext in inputExt:
        for file_path in srcRoot.rglob(f'*{ext}'):
            rel_path = file_path.relative_to(srcRoot)
            group_path = rel_path.parent.as_posix()
            camera_name = rel_path.parts[0]
            grouped_files[group_path].append((file_path, camera_name))
    final_files_to_process = []
    for group_path, files in grouped_files.items():
        files.sort(key=lambda x: str(x[0]))
        shift_val = 0
        group_parts = Path(group_path).parts 
        for resync_path, shift in resync_dict.items():
            resync_parts = Path(resync_path).parts 
            if len(resync_parts) <= len(group_parts):
                if group_parts[-len(resync_parts):] == resync_parts:
                    shift_val = shift
                    break
        if shift_val:
            shift_val %= len(files)
            files = files[shift_val:] + files[:shift_val] 
        valid_count = (len(files) // ppr) * ppr
        if valid_count == 0:
            print(colored(f"Warning ⚠️: {group_path} scartato (file totali minori del PPR: {ppr}).", 'yellow'))
            continue
        files = files[:valid_count]
        final_files_to_process.extend([(f_path, group_path, cam) for f_path, cam in files])
    outputRoot = dstRoot / outputName
    group_counters = defaultdict(int)
    with h5py.File(outputRoot, 'w') as h5file:
        for full_path, group_path, camera_name in tqdm(final_files_to_process, desc="Generazione HDF5"):
            try:
                img = cv.imread(str(full_path), cv.IMREAD_GRAYSCALE)
                if img is None: continue
                rotation_angle = getattr(task_config.Settings.Rotation, camera_name, None)
                if rotation_angle is not None:
                    img = cv.rotate(img, rotation_angle)
                group = h5file.require_group(group_path)
                name = f"{group_counters[group_path]:05d}"
                group.create_dataset(name, data=img)
                group_counters[group_path] += 1      
            except Exception as e:
                print(colored(f"Errore nell'elaborazione di {full_path}: {e}", 'red'))
    return
#%% Defining Main Function
def main(Config):
    task_config = Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1
    if task_config.General.Activation is True:
        print('.... Task1:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        srcRoot = (main_root / 
                   Config.Paths.DataRoots.ResourcesRoot / 
                   Config.Paths.DataRoots.RawDataRoot / 
                   Config.Paths.DataRoots.CaseStudyRoot())
        dstRoot = (main_root / 
                   Config.Paths.DataRoots.ResourcesRoot / 
                   Config.Paths.DataRoots.StreamRoot / 
                   Config.Paths.DataRoots.CaseStudyRoot() / 
                   Config.Packages.Drivers.__name__ / 
                   Config.Packages.Drivers.Phases.Phase0.__name__ / 
                   Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ / 
                   task_config.__name__)  
        directoryExplorer(Config, srcRoot, dstRoot)
        print('.... Task1:', colored('Executed ✅', 'green'))
    elif task_config.General.Activation is False:
        print('.... Task1:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task1 Switch (on/off) ❌')
    return