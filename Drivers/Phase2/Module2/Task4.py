#%% Importing Libraries
import numpy as np
from pathlib import Path
import pandas as pd
import open3d as o3d
from termcolor import colored
#%% Defining Main Function
def main(Config):    
    task_conf = Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4
    if task_conf.General.Activation is True:
        print('.... Task4:', colored('Running ℹ️', 'cyan'))
        main_root = Path(Config.Paths.mainRooot)
        try:
            main_root = Path(Config.Paths.mainRooot)
            srcRoot = (
                main_root / 
                Config.Paths.DataRoots.ResourcesRoot / 
                Config.Paths.DataRoots.StreamRoot / 
                Config.Paths.DataRoots.CaseStudyRoot() /
                Config.Packages.Drivers.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.OutputExt)
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
            dstRoot = (
                main_root / 
                Config.Paths.DataRoots.ResourcesRoot / 
                Config.Paths.DataRoots.StreamRoot / 
                Config.Paths.DataRoots.CaseStudyRoot() /
                Config.Packages.Drivers.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.__name__ / 
                Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task4.MetaData.OutputExt)
            pts, areas = [pd.read_pickle(srcRoot)[k] for k in ['pts', 'areas']]
            totalPoses = pd.read_pickle(poseRoot)['ForwardBlade']
            singlePoses = pd.read_pickle(poseRoot)['ForwardSync']
            totalPoints = {
                ph:[np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).transform(p).points)
                for p in totalPoses[ph]] for ph in totalPoses}
            singlePoints = {
                ph: np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).transform(singlePoses[ph]).points)
                for ph in singlePoses}
            pd.to_pickle({'total': totalPoints, 'single': singlePoints, 'areas': areas}, dstRoot)
            print('.... Task4:', colored('Executed ✅', 'green'))
        except Exception as e:
            print('.... Task4:', colored(f'Error: {e} ❌', 'red'))
            raise e
    elif task_conf.General.Activation is False:
        print('.... Task4:', colored('Offline ⚠️', 'yellow'))
    else:
        raise ValueError('Please Set the Task4 Switch (on/off) ❌')
    return 