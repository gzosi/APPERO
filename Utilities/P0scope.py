from Config import Config
import h5py
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

class Settings:
    class Src:
        Database = 'Database3'
        Dataset = 'Dataset4'
        Foreground = 'Foreground'
        Background = 'Background'
    Bounds = [[0, 45]] 

main_root = Path(Config.Paths.mainRooot)
srcRoot = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase0.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.__name__ /
    Config.Packages.Drivers.Phases.Phase0.Modules.Module2.Tasks.Task1.MetaData.OutputName)
settings = Settings
ppr = Config.Settings.Acquisition.PPR
blades = Config.Settings.Acquisition.Blades
period = ppr / blades 
bounds = settings.Bounds
window_name = 'Visualizzatore Stereo (Premi q per uscire)'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720) 
with h5py.File(srcRoot, 'r') as f:
    cameras = list(f.keys())
    group1, group2 = [
        f[camera][settings.Src.Database][settings.Src.Dataset][settings.Src.Foreground]
        for camera in cameras]
    proces = [k for k in group1.keys()
        if any(lower <= (int(k) % period) <= upper for lower, upper in bounds)]
    for key in proces:
        img1 = group1[key][:].astype(np.uint8)
        img2 = group2[key][:].astype(np.uint8)
        combined = np.hstack((img1, img2))
        if len(combined.shape) == 2:
            combined = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            img=combined, 
            text=f"Key: {key}", 
            org=(20, 40), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1.0, 
            color=(0, 255, 0), 
            thickness=2, 
            lineType=cv2.LINE_AA)
        cv2.imshow(window_name, combined)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            print("Visualizzazione interrotta dall'utente.")
            break
cv2.destroyAllWindows()