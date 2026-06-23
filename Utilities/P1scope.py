from Config import Config 
from pathlib import Path
main_root = Path(Config.Paths.mainRooot)
import pandas as pd
root = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase1.__name__ /
    Config.Packages.Drivers.Phases.Phase1.Modules.Module3.__name__ /
    Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.__name__ /
    Config.Packages.Drivers.Phases.Phase1.Modules.Module3.Tasks.Task2.MetaData.OutputExt)
data = pd.read_pickle(root)['Dataset1'][('Camera1', 'Camera2')]
ret = data['Model27'].loc['Ret']
k1 = data['Model27'].loc['K1']
k2 = data['Model27'].loc['K2']
print('Resti della calibrazione stereo')
print(ret)
print('Matrici interne')
print(k1)
print('-----------------')
print(k2)