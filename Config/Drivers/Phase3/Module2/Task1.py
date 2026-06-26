import numpy as np
#%% Defining Config Packet
class Task1:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Calib:
            Dataset = 'Dataset1'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        class InnerShell:
            occupancyLimit = 2
            class Island:
                MaxDist = 0.5
            class Normals:
                CameraPose = (0.0, 0.0, 0.0) 
                Radius = 100
                Max_nn = 50                 
                TileSize = 2.5
        class ControlShell:
            k_neighbors = 10
            dilation_iters = 5
            smooth_iters = 10
        class OuterShell:
            class Boundary:
                k_neighbors = 75
                min_angle_deg = 115
                Threshold = 0.5
            class Volume:
                voxelSize = 0.5
                rbfKernel = 'thin_plate_spline'
                rbfSmoothing = 2.5
                maxLateralDist = 0.5
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0