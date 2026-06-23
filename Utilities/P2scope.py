from Config import Config 
from pathlib import Path
import pyvista as pv
main_root = Path(Config.Paths.mainRooot)
import pandas as pd
root = (main_root /
    Config.Paths.DataRoots.ResourcesRoot /
    Config.Paths.DataRoots.StreamRoot / 
    Config.Paths.DataRoots.CaseStudyRoot() /
    Config.Packages.Drivers.__name__ /
    Config.Packages.Drivers.Phases.Phase2.__name__ /
    Config.Packages.Drivers.Phases.Phase2.Modules.Module2.__name__ /
    Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.__name__ /
    Config.Packages.Drivers.Phases.Phase2.Modules.Module2.Tasks.Task3.MetaData.OutputExt)
data = pd.read_pickle(root)['pts']
plotter = pv.Plotter()
plotter.add_points(
    data, 
    color='red', 
    point_size=2.0,
    render_points_as_spheres=True)
plotter.show_grid()
plotter.show_axes()
plotter.show()